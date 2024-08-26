import numpy as np
import pandas as pd


def match_gen_load_names(list1, list2):
    """Match generator and load names"""
    list1.sort()
    list2.sort()
    prefix_dict = {}
    for item in list2:
        prefix = item.split("_")[0]
        if prefix not in prefix_dict:
            prefix_dict[prefix] = []
        prefix_dict[prefix].append(item)

    result = {}
    for item in list1:
        prefix = item.split("_")[0]
        if prefix in prefix_dict and prefix_dict[prefix]:
            result[item] = prefix_dict[prefix].pop(0)
        else:
            print(f"No match found for {item}")

    return result


def make_storage_resources(data):
    sced_gen = data["sced_gen_resource"][
        ["Resource Name", "QSE", "DME", "Resource Type"]
    ].drop_duplicates()
    sced_gen_storage_names = sced_gen[sced_gen["Resource Type"] == "PWRSTR"][
        "Resource Name"
    ].unique()
    sced_load_all = data["sced_load_resource"]["Resource Name"].unique()
    matched_load_gen_names = match_gen_load_names(sced_gen_storage_names, sced_load_all)

    storage_resources = (
        pd.DataFrame(
            {
                "gridstatus_id": list(matched_load_gen_names.keys()),
                "ercot_gen_resource_name": list(matched_load_gen_names.keys()),
                "ercot_load_resource_name": list(matched_load_gen_names.values()),
            },
        )
        .merge(
            data["settlement_point_mapping"],
            how="left",
            left_on="ercot_gen_resource_name",
            right_on="Resource Name",
        )
        .drop(columns=["Resource Name"])
        .merge(
            sced_gen,
            how="left",
            left_on="ercot_gen_resource_name",
            right_on="Resource Name",
        )
        .drop(columns=["Resource Name"])
        .rename(columns={"Settlement Point Name": "settlement_point_name"})
    )

    # Fill missing settlement point names with manual matches
    manual_matches = {
        "ESTONIAN_BES1": "ESTONIAN_ALL",
        "FENCESLR_BESS1": "FENCESLR_ALL",
        "MV_VALV4_BESS": "MV_VALV4_RN",
        "RVRVLYS_ESS1": "RVRVLYS_ALL",
        "RVRVLYS_ESS2": "RVRVLYS_ALL",
        "WFTANK_ESS1": "WFTANK_ESS1",
        "LONESTAR_BESS": "LONESTAR_RN",
    }
    storage_resources["settlement_point_name"] = storage_resources[
        "settlement_point_name"
    ].fillna(storage_resources["ercot_gen_resource_name"].map(manual_matches))

    # Get SARA data and merge with storage_resources
    cols = [
        "gridstatus_id",
        "unit_name",
        "ercot_gen_resource_name",
        "ercot_load_resource_name",
        "settlement_point_name",
        "qse",
        "dme",
        "resource_type",
        "county",
        "zone",
        "in_service_year",
        "installed_capacity_rating",
        "summer_capacity_mw",
        "generation_interconnection_project_code",
    ]
    storage_resources = (
        storage_resources.merge(
            data["sara"],
            how="left",
            left_on="ercot_gen_resource_name",
            right_on="Unit Code",
        )
        .drop(columns=["Unit Code", "Fuel", "New Planned Project Additions to Report"])
        .rename(
            columns={
                "Unit Name": "unit_name",
                "County": "county",
                "Zone": "zone",
                "Generation Interconnection Project Code": "generation_interconnection_project_code",  # noqa
                "In Service Year": "in_service_year",
                "Installed Capacity Rating": "installed_capacity_rating",
                "Summer Capacity (MW)": "summer_capacity_mw",
                "QSE": "qse",
                "DME": "dme",
                "Resource Type": "resource_type",
            },
        )[cols]
    )

    return storage_resources


def extract_curve(df, curve_name):
    mw_cols = [x for x in df.columns if x.startswith((curve_name + "-MW",
                                                      curve_name + " MW"))]
    price_cols = [x for x in df.columns if x.startswith((curve_name + "-Price",
                                                         curve_name + " Price"))]

    if len(mw_cols) == 0 or len(price_cols) == 0:
        return np.nan

    def combine_mw_price(row):
        return [
            (mw, price)
            for mw, price in zip(row[mw_cols], row[price_cols])
            if pd.notnull(mw) and pd.notnull(price)
        ]

    # round price columns to 2 decimal places
    df[price_cols] = df[price_cols].round(2)
    return df.apply(combine_mw_price, axis=1)


def extract_as_curve(df, product_name):
    """
    Extract ancillary curves, which are grouped by quantity and have a
    different pattern
    """
    mw_cols = [x for x in df.columns if x.startswith("QUANTITY MW")]
    price_cols = [x for x in df.columns if x.endswith(product_name)]

    if len(mw_cols) == 0 or len(price_cols) == 0:
        return np.nan

    def combine_mw_price(row):
        return [
            (mw, price)
            for mw, price in zip(row[mw_cols], row[price_cols])
            if pd.notnull(mw) and pd.notnull(price)
        ]

    # round price columns to 2 decimal places
    df[price_cols] = df[price_cols].round(2)
    return df.apply(combine_mw_price, axis=1)


def process_dam_gen(df):
    time_cols = [
        "Interval Start",
        "Interval End",
    ]

    resource_cols = [
        "QSE",
        "DME",
        "Resource Name",
        "Resource Type",
        "Settlement Point Name",
    ]

    telemetry_cols = [
        "Resource Status",
        "HSL",
        "LSL",
        "Start Up Hot",
        "Start Up Inter",
        "Start Up Cold",
        "Min Gen Cost",
    ]

    energy_award_cols = [
        "Awarded Quantity",
        "Energy Settlement Point Price",
    ]

    as_cols = [
        "RegUp Awarded",
        "RegUp MCPC",
        "RegDown Awarded",
        "RegDown MCPC",
        "RRSPFR Awarded",
        "RRSFFR Awarded",
        "RRSUFR Awarded",
        "RRS MCPC",
        "ECRSSD Awarded",
        "ECRS MCPC",
        "NonSpin Awarded",
        "NonSpin MCPC",
    ]

    curve = "QSE submitted Curve"

    df[curve] = extract_curve(df, "QSE submitted Curve")

    all_cols = resource_cols + telemetry_cols + energy_award_cols + as_cols + [curve]

    for col in all_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[time_cols + all_cols]

    return df


def process_dam_load(df):
    time_cols = [
        "Time",
        "Interval Start",
        "Interval End",
    ]

    resource_cols = ["Load Resource Name"]

    telemetry_cols = [
        "Max Power Consumption for Load Resource",
        "Low Power Consumption for Load Resource",
    ]

    as_cols = [
        "RegUp Awarded",
        "RegUp MCPC",
        "RegDown Awarded",
        "RegDown MCPC",
        "RRSPFR Awarded",
        "RRSFFR Awarded",
        "RRSUFR Awarded",
        "RRS MCPC",
        "ECRSSD Awarded",
        "ECRSMD Awarded",
        "ECRS MCPC",
        "NonSpin Awarded",
        "NonSpin MCPC",
    ]

    all_cols = resource_cols + telemetry_cols + as_cols

    for col in all_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[time_cols + all_cols]

    # rename for consistency
    # with gen columns
    df = df.rename(
        columns={
            "Load Resource Name": "Resource Name",
        },
    )

    return df


_AS_COMMON = [
    "Time",
    "Interval Start",
    "Interval End",
    "QSE",
    "DME",
    "Resource Name",
    "Multi-Hour Block Flag",
    ]


_AS_PRODUCTS = [
    "RRSPFR",
    "RRSFFR",
    "RRSUFR",
    "ECRS",
    "OFFEC",
    "ONLINE NONSPIN",
    "REGUP",
    "REGDOWN",
    "OFFLINE NONSPIN",
    ]


def process_dam_as_curves(df):
    """
    Handle the special case of
    """
    for product in _AS_PRODUCTS:
        crv = extract_as_curve(df, product)
        col_name = f"{product} Offer Curve"
        df[col_name] = crv
    return df


def dedup_as_offers(df):
    """
    Offers from a resource can be spread over multiple lines (or at least two)
    with different products on the different lines even though the format can
    accomodate all in one line.  Collapse them.
    """

    grp = df.groupby(["Interval End", "Resource Name"])
    sum_cols = [c for c in df.columns if "PRICE" in c]
    # first_cols = set(df.columns).difference(sum_cols)
    rules = {c: "first" for c in df.columns}
    rules.update({c: "sum" for c in df.columns if c in sum_cols})
    df_fixed = grp.aggregate(rules)
    df_fixed.reset_index(drop=True, inplace=True)
    return df_fixed


def process_dam_load_as_offers(df):
    if "QSE" not in df.columns:
        # after Interval End
        index = df.columns.tolist().index("Interval End") + 1
        df.insert(index, "QSE", np.nan)

    if "DME" not in df.columns:
        # after QSE
        index = df.columns.tolist().index("QSE") + 1
        df.insert(index, "DME", np.nan)

    df = df.rename(
        columns={
            "Load Resource Name": "Resource Name",
        },
    )

    df = dedup_as_offers(df)

    df = process_dam_as_curves(df)

    keep_columns = _AS_COMMON + [f"{p} Offer Curve" for p in _AS_PRODUCTS]
    df = df[keep_columns]

    return df


def process_dam_gen_as_offers(df):
    if "QSE" not in df.columns:
        # after Interval End
        index = df.columns.tolist().index("Interval End") + 1
        df.insert(index, "QSE", np.nan)

    if "DME" not in df.columns:
        # after QSE
        index = df.columns.tolist().index("QSE") + 1
        df.insert(index, "DME", np.nan)

    df = df.rename(
        columns={
            "Generation Resource Name": "Resource Name",
        },
    )

    df_fixed = dedup_as_offers(df)

    df = process_dam_as_curves(df_fixed)

    keep_columns = _AS_COMMON + [f"{p} Offer Curve" for p in _AS_PRODUCTS]
    df = df[keep_columns]

    return df


def process_dam_energy_only(df, bid_or_offer="Bid"):

    df = df.rename(columns={"Energy Only Offer ID": "Offer ID",
                            "Energy Only Bid ID": "Bid ID"})

    curve = f"Energy Only {bid_or_offer} Curve"

    df[curve] = extract_curve(df, f"Energy Only {bid_or_offer}")

    to_drop = [c for c in df.columns if "MW" in c or "Price" in c]

    df = df.drop(to_drop, axis=1)

    return df


def process_sced_gen(df, process_sced2=False):
    time_cols = [
        "Interval Start",
        "Interval End",
        "SCED Time Stamp",
    ]

    resource_cols = ["QSE", "DME", "Resource Name", "Resource Type"]

    telemetry_cols = [
        "Telemetered Resource Status",
        "Output Schedule",
        "HSL",
        "HASL",
        "HDL",
        "LSL",
        "LASL",
        "LDL",
        "Base Point",
        "Telemetered Net Output ",
    ]

    as_cols = [
        "Ancillary Service REGUP",
        "Ancillary Service REGDN",
        "Ancillary Service RRS",
        "Ancillary Service RRSFFR",
        "Ancillary Service NSRS",
        "Ancillary Service ECRS",
    ]

    tpo_cols = [
        "Start Up Cold Offer",
        "Start Up Hot Offer",
        "Start Up Inter Offer",
        "Min Gen Cost",
        "SCED TPO Offer Curve",
    ]

    sced1_offer_col = "SCED1 Offer Curve"
    sced2_offer_col = "SCED2 Offer Curve"
    sced_offer_cols = [sced1_offer_col]

    df[sced1_offer_col] = extract_curve(df, "SCED1 Curve")
    if process_sced2_curve:
        df[sced2_offer_col] = extract_curve(df, "SCED2 Curve")
        sced_offer_cols  += [sced2_offer_col]

    df[tpo_cols[-1]] = extract_curve(df, "Submitted TPO")

    all_cols = resource_cols + telemetry_cols + as_cols + sced_offer_cols + tpo_cols

    for col in all_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[time_cols + all_cols]

    # standardized to same naming as load
    # clean up column names
    df = df.rename(
        columns={
            "Ancillary Service RRS": "AS Responsibility for RRS",
            "Ancillary Service RRSFFR": "AS Responsibility for RRSFFR",
            "Ancillary Service NSRS": "AS Responsibility for NonSpin",
            "Ancillary Service REGUP": "AS Responsibility for RegUp",
            "Ancillary Service REGDN": "AS Responsibility for RegDown",
            "Ancillary Service ECRS": "AS Responsibility for ECRS",
            # remove space
            "Telemetered Net Output ": "Telemetered Net Output",
        },
    )

    return df


def process_sced_load(df):
    time_cols = [
        "Interval Start",
        "Interval End",
        "SCED Time Stamp",
    ]

    resource_cols = ["QSE", "DME", "Resource Name"]

    telemetry_cols = [
        "Telemetered Resource Status",
        "Max Power Consumption",
        "Low Power Consumption",
        "Real Power Consumption",
        "HASL",
        "HDL",
        "LASL",
        "LDL",
        "Base Point",
    ]

    as_cols = [
        "AS Responsibility for RRS",
        "AS Responsibility for RRSFFR",
        "AS Responsibility for NonSpin",
        "AS Responsibility for RegUp",
        "AS Responsibility for RegDown",
        "AS Responsibility for ECRS",
    ]

    bid_curve_col = "SCED Bid to Buy Curve"

    df[bid_curve_col] = extract_curve(df, "SCED Bid to Buy Curve")

    all_cols = resource_cols + telemetry_cols + as_cols + [bid_curve_col]
    for col in all_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[time_cols + all_cols]

    return df


# # backup for more node names
# pd.read_html("https://www.ercot.com/content/cdr/html/current_np6788.html", skiprows=3)[0][0] # noqa
# # todo add in QSE
# # todo prefix and county match
