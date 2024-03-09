# Import package
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Cleaned-Data-FY22-24.csv')

################## USE SELECTED COLUMNS ##################

columns = ['contract_transaction_unique_key',
    'potential_total_value_of_award', 
    # 'awarding_office_name', 
    # 'funding_office_name', 
    'foreign_funding', 
    # 'recipient_city_name', 
    # 'recipient_county_name', 
    # 'recipient_state_code', 
    # 'recipient_zip_4_code', 
    # 'primary_place_of_performance_city_name', 
    'prime_award_transaction_place_of_performance_county_fips_code', 
    'prime_award_transaction_place_of_performance_state_fips_code', 
    # 'primary_place_of_performance_zip_4',     
    'award_type', 
    'type_of_contract_pricing', 
    # 'inherently_governmental_functions_description', 
    # 'product_or_service_code', 
    'contract_bundling', 
    'naics_code', 
    'recovered_materials_sustainability', 
    'domestic_or_foreign_entity', 
    'epa_designated_product', 
    'country_of_product_or_service_origin', 
    'place_of_manufacture', 
    # 'subcontracting_plan', 
    'extent_competed', 
    'solicitation_procedures', 
    'type_of_set_aside', 
    'number_of_offers_received', 
    'commercial_item_acquisition_procedures', 
    'small_business_competitiveness_demonstration_program', 
    'simplified_procedures_for_certain_commercial_items', 
    'local_area_set_aside', 
    'clinger_cohen_act_planning', 
    'labor_standards', 
    'contracting_officers_determination_of_business_size', 
    'labor_surplus_area_firm', 'contingency_humanitarian_or_peacekeeping_operation', 
    'alaskan_native_corporation_owned_firm', 
    'american_indian_owned_business', 
    'indian_tribe_federally_recognized', 
    'native_hawaiian_organization_owned_firm', 
    'tribally_owned_firm', 
    'veteran_owned_business', 
    'service_disabled_veteran_owned_business', 
    'woman_owned_business', 
    'women_owned_small_business', 
    'economically_disadvantaged_women_owned_small_business', 
    'joint_venture_women_owned_small_business', 
    'joint_venture_economic_disadvantaged_women_owned_small_bus', 
    'minority_owned_business', 
    'subcontinent_asian_asian_indian_american_owned_business', 
    'asian_pacific_american_owned_business', 
    'black_american_owned_business', 
    'hispanic_american_owned_business', 
    'native_american_owned_business', 
    'other_minority_owned_business', 
    'contracting_officers_determination_of_business_size_code', 
    'emerging_small_business', 
    'community_developed_corporation_owned_firm', 
    'us_federal_government', 
    'federally_funded_research_and_development_corp', 
    'federal_agency', 
    'us_state_government', 
    'us_local_government', 
    'city_local_government', 
    'county_local_government', 
    'inter_municipal_local_government', 
    'local_government_owned', 
    'municipality_local_government', 
    'school_district_local_government', 
    'township_local_government', 
    'us_tribal_government', 
    'foreign_government', 
    'organizational_type', 
    'corporate_entity_not_tax_exempt', 
    'corporate_entity_tax_exempt', 
    'partnership_or_limited_liability_partnership', 
    'sole_proprietorship', 
    'small_agricultural_cooperative', 
    'international_organization', 
    'us_government_entity', 
    'community_development_corporation', 
    'domestic_shelter', 
    'educational_institution', 
    'foundation', 
    'hospital_flag', 
    'manufacturer_of_goods', 
    'veterinary_hospital', 
    'hispanic_servicing_institution', 
    'receives_contracts', 
    'receives_financial_assistance', 
    'receives_contracts_and_financial_assistance', 
    'airport_authority', 
    'council_of_governments', 
    'housing_authorities_public_tribal', 
    'interstate_entity', 
    'planning_commission', 
    'port_authority', 
    'transit_authority', 
    'subchapter_scorporation', 
    'limited_liability_corporation', 
    'foreign_owned', 
    'for_profit_organization', 
    'nonprofit_organization', 
    'other_not_for_profit_organization', 
    'the_ability_one_program', 
    'private_university_or_college', 
    'state_controlled_institution_of_higher_learning', 
    '1862_land_grant_college', 
    '1890_land_grant_college', 
    '1994_land_grant_college', 
    'minority_institution', 
    'historically_black_college', 
    'tribal_college', 
    'alaskan_native_servicing_institution', 
    'native_hawaiian_servicing_institution', 
    'school_of_forestry', 
    'veterinary_college', 
    'dot_certified_disadvantage', 
    'self_certified_small_disadvantaged_business', 
    'small_disadvantaged_business', 
    'c8a_program_participant', 
    'historically_underutilized_business_zone_hubzone_firm', 
    'sba_certified_8a_joint_venture', 'action_date', 'contract_award_unique_key','period_of_performance_start_date',
    'number_of_offers_received'
]
df = df[columns]

################## HANDLE MISSING VALUES ##################

# retain columns with less than 20% missing values
threshold = len(df) * 0.2
cols_to_use = df.columns[df.isna().sum() <= threshold]
df_drop = df[cols_to_use]

################## REMOVE DUPLICATES ##################

# Change the action_date column from object to datetime
df_drop.loc[:,'action_date'] = pd.to_datetime(df_drop.loc[:,'action_date'])

# remove rows with the same contract award unique key and only keep the one with the latest action date
no_dup_df = df_drop.sort_values(['contract_award_unique_key','action_date'], ascending= [True,False]).drop_duplicates(subset="contract_award_unique_key", keep= "first")
no_dup_df['contract_award_unique_key']

no_dup_df['period_of_performance_start_date'] = pd.to_datetime(no_dup_df['period_of_performance_start_date'])
# Reset the index for better order
fiscal_df= no_dup_df.reset_index().drop("index", axis = 1)

################## HANDLE OUTLIERS ##################

seventy_fifth = fiscal_df["potential_total_value_of_award"].quantile(0.75) 
twenty_fifth = fiscal_df["potential_total_value_of_award"].quantile(0.25) 
iqr = fiscal_df["potential_total_value_of_award"].quantile(0.75) - fiscal_df["potential_total_value_of_award"].quantile(0.25)

right_off = seventy_fifth + iqr*1.5
left_off = twenty_fifth - iqr*1.5
fiscal_df = fiscal_df[(fiscal_df["potential_total_value_of_award"] <= right_off) & (fiscal_df["potential_total_value_of_award"] > left_off)]

################## DROP UNIMPORTANT COLUMNS ##################

# remove unimportant columns
fiscal_df.drop([
    "contract_transaction_unique_key",
    "contract_award_unique_key",
    'action_date',
    'period_of_performance_start_date'
], axis = 1, inplace=True)

################## ENCODING ##################

# this should have been categorical
fiscal_df['naics_code'] = fiscal_df['naics_code'].astype("object")

cat_df = fiscal_df.select_dtypes('O')

# detect colums that are true-false values
columns_with_f = cat_df.apply(lambda col: 'f' in col.values)
columns_containing_f = cat_df.columns[columns_with_f]
filtered_df = cat_df[columns_containing_f]

# binary encoding for true-false columns
for column in filtered_df.columns:
    fiscal_df[column] = (fiscal_df[column] == "t").astype(int)

# final data before ML
fiscal_df_cat = fiscal_df.select_dtypes('O').columns
encoded_df = pd.get_dummies(fiscal_df, columns=fiscal_df_cat, drop_first= True)
machine_df = encoded_df.dropna(axis=1)
machine_df

machine_df.to_csv('machine_df.csv', index=False)