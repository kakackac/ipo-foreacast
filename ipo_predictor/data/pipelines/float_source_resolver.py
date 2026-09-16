"""Select a dated, price-linked float disclosure without reviving superseded values."""
import math
import pandas as pd

FLOAT_FIELDS = (
    "public_float_shares", "public_float_ratio_disclosed", "public_float_parse_method",
    "public_float_parse_evidence", "total_post_listing_shares",
    "total_post_listing_shares_parse_method", "total_post_listing_shares_parse_evidence",
    "total_post_listing_shares_parser_validation_status",
)


def resolve_public_float(documents, offering_price, listing_date):
    result = {field: None for field in FLOAT_FIELDS if field.startswith("public_float_")}
    result.update(public_float_rcept_no=None, public_float_rcept_dt=None,
                  public_float_resolution_status="no_verified_source")
    cutoff = pd.Timestamp(listing_date)
    ordered = sorted(documents, key=lambda pair: (
        str(pair[0].get("rcept_dt", "")), str(pair[0].get("rcept_no", ""))), reverse=True)
    for metadata, document in ordered:
        published = pd.to_datetime(metadata.get("rcept_dt"), errors="coerce")
        if pd.isna(published) or published >= cutoff:
            continue
        method = document.get("public_float_parse_method")
        if method == "conflicting_float_ratios_review_required":
            result.update(public_float_parse_method=method,
                          public_float_parse_evidence=document.get("public_float_parse_evidence"),
                          public_float_resolution_status="latest_conflict_blocks_older_fallback")
            return result
        if method not in ("disclosed_public_float_ratio_direct_context",
                          "disclosed_public_float_shares_direct_context"):
            continue
        # Prospectus candidates are discovered by corp_code; price linkage is also required.
        if metadata.get("supplementary"):
            try:
                matches = math.isfinite(float(offering_price)) and float(offering_price) > 0 and (
                    float(document.get("offering_price")) == float(offering_price))
            except (TypeError, ValueError):
                matches = False
            if not matches:
                result["public_float_resolution_status"] = "latest_source_price_unverified"
                return result
        if method == "disclosed_public_float_shares_direct_context" and document.get(
            "total_post_listing_shares_parser_validation_status"
        ) != "structurally_verified":
            result["public_float_resolution_status"] = "latest_source_denominator_unverified"
            return result
        result.update({field: document.get(field) for field in FLOAT_FIELDS})
        result.update(public_float_rcept_no=str(metadata["rcept_no"]),
                      public_float_rcept_dt=published,
                      public_float_resolution_status="latest_disclosed_source_selected")
        return result
    return result
