"""
Multi-Blocking and Candidate Generation Module using Polars.
Generates lightweight candidate pairs (s1_id, cand_id) using country-partitioned blocking keys.
"""
import gc
import time
import polars as pl
from typing import Dict, List, Set, Tuple

from src.config import LEGAL_RE

def get_lean_blocking_keys(lazy_df: pl.LazyFrame, country_str: str) -> pl.LazyFrame:
    """
    Generates vectorized blocking keys and preserves text fields for records in country_str.
    """
    return lazy_df.filter(pl.col("country") == country_str).with_columns([
        pl.col("business_name").str.to_lowercase().str.replace_all(r"[^a-z0-9\s]", " ").str.replace_all(LEGAL_RE, " ").str.replace_all(r"\s+", " ").str.strip_chars().alias("norm_name"),
        pl.col("business_address").str.extract_all(r"\b\d+\b").alias("addr_digits"),
        pl.col("business_address").str.extract_all(r"\b\d{5,6}\b").alias("postal_codes"),
    ]).with_columns([
        pl.col("norm_name").str.split(" ").list.first().alias("tok1"),
        pl.col("norm_name").str.split(" ").list.get(1, null_on_oob=True).alias("tok2"),
        pl.col("norm_name").str.split(" ").list.get(2, null_on_oob=True).alias("tok3"),
        pl.col("norm_name").str.slice(0, 5).alias("pref5"),
        pl.col("addr_digits").list.first().alias("house_num"),
        pl.col("postal_codes").list.first().alias("postal_code"),
    ]).select(["entity_id", "business_name", "business_address", "norm_name", "tok1", "tok2", "tok3", "pref5", "house_num", "postal_code"])

class VectorizedBlocker:
    """
    Fast country-partitioned candidate generator.
    """
    def __init__(self, tok12_max_count: int = 400, tok1_max_count: int = 80):
        self.tok12_max_count = tok12_max_count
        self.tok1_max_count = tok1_max_count

    def generate_candidates_from_keys(self, s1_k: pl.DataFrame, s23_k: pl.DataFrame) -> pl.DataFrame:
        """
        Generates candidates for pre-collected s1_k and s23_k DataFrames.
        """
        if len(s1_k) == 0 or len(s23_k) == 0:
            return pl.DataFrame(schema={
                "s1_id": pl.Utf8, "cand_id": pl.Utf8,
                "name1": pl.Utf8, "addr1": pl.Utf8,
                "name2": pl.Utf8, "addr2": pl.Utf8
            })

        out_cols = [
            pl.col("entity_id").alias("s1_id"),
            pl.col("entity_id_cand").alias("cand_id"),
            pl.col("business_name").alias("name1"),
            pl.col("business_address").alias("addr1"),
            pl.col("business_name_cand").alias("name2"),
            pl.col("business_address_cand").alias("addr2"),
        ]

        # 1. Exact norm name match
        j1 = s1_k.filter(pl.col("norm_name") != "").join(
            s23_k, on="norm_name", suffix="_cand"
        ).select(out_cols)

        # 2. Tok1 + House Number
        j2 = s1_k.filter((pl.col("tok1").is_not_null()) & (pl.col("house_num").is_not_null()) & (pl.col("tok1").str.len_chars() >= 3)).join(
            s23_k, on=["tok1", "house_num"], suffix="_cand"
        ).select(out_cols)

        # 3. Tok1 + Postal Code
        j3 = s1_k.filter((pl.col("tok1").is_not_null()) & (pl.col("postal_code").is_not_null()) & (pl.col("tok1").str.len_chars() >= 3)).join(
            s23_k, on=["tok1", "postal_code"], suffix="_cand"
        ).select(out_cols)

        # 4. Pref5 + House Number
        j4 = s1_k.filter((pl.col("pref5").is_not_null()) & (pl.col("house_num").is_not_null()) & (pl.col("pref5").str.len_chars() >= 4)).join(
            s23_k, on=["pref5", "house_num"], suffix="_cand"
        ).select(out_cols)

        # 5. Capped Tok1 + Tok2
        s23_tok12_counts = s23_k.filter((pl.col("tok1").is_not_null()) & (pl.col("tok2").is_not_null()) & (pl.col("tok1").str.len_chars() >= 3) & (pl.col("tok2").str.len_chars() >= 3)).group_by(["tok1", "tok2"]).agg(pl.len().alias("k_cnt")).filter(pl.col("k_cnt") <= self.tok12_max_count)
        
        s23_capped_tok12 = s23_k.join(s23_tok12_counts, on=["tok1", "tok2"])
        s1_capped_tok12 = s1_k.join(s23_tok12_counts, on=["tok1", "tok2"])
        
        j5 = s1_capped_tok12.join(
            s23_capped_tok12, on=["tok1", "tok2"], suffix="_cand"
        ).select(out_cols)

        # 6. Rare Tok1
        s23_tok1_counts = s23_k.filter((pl.col("tok1").is_not_null()) & (pl.col("tok1").str.len_chars() >= 4)).group_by("tok1").agg(pl.len().alias("k_cnt")).filter(pl.col("k_cnt") <= self.tok1_max_count)
        
        s23_capped_tok1 = s23_k.join(s23_tok1_counts, on="tok1")
        s1_capped_tok1 = s1_k.join(s23_tok1_counts, on="tok1")
        
        j6 = s1_capped_tok1.join(
            s23_capped_tok1, on="tok1", suffix="_cand"
        ).select(out_cols)

        cands_c = pl.concat([j1, j2, j3, j4, j5, j6]).unique(subset=["s1_id", "cand_id"])
        del s23_capped_tok12, s1_capped_tok12, s23_capped_tok1, s1_capped_tok1, j1, j2, j3, j4, j5, j6
        return cands_c

    def generate_candidates(self, l_s1: pl.LazyFrame, l_s2: pl.LazyFrame, l_s3: pl.LazyFrame, countries: List[str]) -> pl.DataFrame:
        """
        Generates candidate pairs DataFrame with schema: [s1_id, cand_id, name1, addr1, name2, addr2]
        """
        start_t = time.time()
        print(f"[Blocker] Generating candidate pairs across {len(countries)} countries...", flush=True)
        
        all_cand_dfs = []
        
        for c in countries:
            c_start = time.time()
            s1_k = get_lean_blocking_keys(l_s1, c).collect()
            if len(s1_k) == 0:
                continue
                
            s2_k = get_lean_blocking_keys(l_s2, c).collect()
            s3_k = get_lean_blocking_keys(l_s3, c).collect()
            
            s23_k = pl.concat([s2_k, s3_k])
            del s2_k, s3_k
            gc.collect()
            
            cands_c = self.generate_candidates_from_keys(s1_k, s23_k)
            all_cand_dfs.append(cands_c)
            
            del s1_k, s23_k
            gc.collect()
            print(f"  Country '{c}' generated {len(cands_c)} candidate pairs in {time.time() - c_start:.2f}s.", flush=True)

        final_cands = pl.concat(all_cand_dfs)
        print(f"[Blocker] Multi-blocking complete! Total candidate pairs: {len(final_cands)} in {time.time() - start_t:.2f}s.", flush=True)
        return final_cands
