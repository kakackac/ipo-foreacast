"""Verify actual stage cutoffs, not merely availability before listing."""
import pandas as pd


def timestamp(value, conservative_date=False):
    if pd.isna(value):
        return pd.NaT
    try:
        parsed = pd.Timestamp(value)
        if parsed.tzinfo is None:
            parsed = parsed.tz_localize("Asia/Seoul")
        if conservative_date and parsed.hour == parsed.minute == parsed.second == parsed.microsecond == 0:
            parsed += pd.Timedelta(days=1)
        return parsed.tz_convert("UTC")
    except (ValueError, TypeError):
        return pd.NaT


def validate_prediction_times(frame, profile, time_audit):
    result = pd.Series(False, index=frame.index, dtype=bool)
    columns = {"event_id", "prediction_at", "prediction_time_source", "demand_result_available_at",
               "demand_result_time_source", "listing_date"}
    if not columns.issubset(frame.columns) or time_audit is None:
        return result
    if not {"event_id", "feature_name", "available_at", "time_validation_status"}.issubset(time_audit.columns):
        return result
    evidence = time_audit.copy()
    evidence["event_id"] = evidence["event_id"].astype(str)
    duplicates = set(evidence.loc[evidence.duplicated(["event_id", "feature_name"], keep=False), "event_id"])
    groups = {key: group.set_index("feature_name") for key, group in evidence.groupby("event_id")}
    for index, row in frame.iterrows():
        event = str(row["event_id"])
        if event in duplicates or event not in groups:
            continue
        if any(pd.isna(row[key]) or not str(row[key]).strip()
               for key in ("prediction_time_source", "demand_result_time_source")):
            continue
        cutoff = timestamp(row["prediction_at"])
        demand = timestamp(row["demand_result_available_at"], conservative_date=True)
        listing = timestamp(row["listing_date"])
        if any(pd.isna(value) for value in (cutoff, demand, listing)) or not cutoff < listing:
            continue
        # Date-only result publication cannot prove a pre-demand cutoff on that day.
        earliest_demand = timestamp(row["demand_result_available_at"])
        if profile.name == "pre_demand" and not cutoff < earliest_demand:
            continue
        if profile.name == "post_demand" and not demand <= cutoff:
            continue
        approved = True
        group = groups[event]
        for feature in profile.feature_names:
            if pd.isna(row.get(feature, pd.NA)):
                continue
            if feature not in group.index:
                approved = False
                break
            observation = group.loc[feature]
            available = timestamp(observation["available_at"], conservative_date=True)
            missing = observation.get("is_missing", True)
            if pd.isna(missing) or missing != False or observation["time_validation_status"] != "pre_listing_verified" or pd.isna(available) or not available < cutoff:
                approved = False
                break
        result.at[index] = approved
    return result
