# Columns to use / code / drop in UKB metadata

columns_to_drop = ['VisitDate_0', 'VisitDate_1', 'VisitDate_2', 'VisitDate_3', 'birthday',  'DBP_2.1',  'SBP_2.1',  'DBP_0.1', 'SBP_0.1',  'PulseRate_0.1', 'PulseRate_2.1',
                   'pwa_heart_rate_2.1', 'pwa_heart_rate_2.2', 'pwa_heart_rate_2.3', 'pwa_heart_rate_2.4',
                   'pwa_brachial_sbp_2.1', 'pwa_brachial_sbp_2.2', 'pwa_brachial_sbp_2.3', 'pwa_brachial_sbp_2.4',
                   'pwa_brachial_dbp_2.1', 'pwa_brachial_dbp_2.2', 'pwa_brachial_dbp_2.3', 'pwa_brachial_dbp_2.4',
                   'pwa_peripheral_pulse_pressure_2.1', 'pwa_peripheral_pulse_pressure_2.2', 'pwa_peripheral_pulse_pressure_2.3', 'pwa_peripheral_pulse_pressure_2.4',
                   'pwa_central_sbp_2.1', 'pwa_central_sbp_2.2', 'pwa_central_sbp_2.3', 'pwa_central_sbp_2.4',
                   'pwa_central_pulse_pressure_2.1', 'pwa_central_pulse_pressure_2.2', 'pwa_central_pulse_pressure_2.3', 'pwa_central_pulse_pressure_2.4',
                   'pwa_number_beats_2.1', 'pwa_number_beats_2.2', 'pwa_number_beats_2.3', 'pwa_number_beats_2.4',
                   'pwa_central_augmentation_pressure_2.1', 'pwa_central_augmentation_pressure_2.2', 'pwa_central_augmentation_pressure_2.3', 'pwa_central_augmentation_pressure_2.4',
                   'pwa_cardiac_output_2.1', 'pwa_cardiac_output_2.2', 'pwa_cardiac_output_2.3', 'pwa_cardiac_output_2.4',
                   'pwa_end_sp_2.1', 'pwa_end_sp_2.2', 'pwa_end_sp_2.3', 'pwa_end_sp_2.4',
                   'pwa_end_sp_index_2.1', 'pwa_end_sp_index_2.2', 'pwa_end_sp_index_2.3', 'pwa_end_sp_index_2.4',
                   'pwa_total_peripheral_resistance_2.1', 'pwa_total_peripheral_resistance_2.2', 'pwa_total_peripheral_resistance_2.3', 'pwa_total_peripheral_resistance_2.4',
                   'pwa_stroke_volume_2.1', 'pwa_stroke_volume_2.2', 'pwa_stroke_volume_2.3', 'pwa_stroke_volume_2.4',
                   'pwa_mean_arterial_pressure_2.1', 'pwa_mean_arterial_pressure_2.2', 'pwa_mean_arterial_pressure_2.3', 'pwa_mean_arterial_pressure_2.4',
                   'pwa_cardiac_index_2.1', 'pwa_cardiac_index_2.2', 'pwa_cardiac_index_2.3', 'pwa_cardiac_index_2.4',
                   'spiro_fvc_0.1', 'spiro_fvc_0.2', 'spiro_pef_0.1', 'spiro_pef_0.2', 
                   'spiro_fvc_2.1', 'spiro_fvc_2.2',  'spiro_qc_0.0',  'spiro_pef_2.1', 'spiro_pef_2.2', 'has_img',
                   'spiro_fvc_0.0', 'spiro_pef_0.0',
                   'Weight', 'BMI_2.0',
                   ]

columns_to_code = ['Sex_0.0', 'Pacemaker_0.0', 'Ethnicity_0.0', 'Long_standing_illness_disability_0.0',  'Long_standing_illness_disability_2.0',
                   'snoring_0.0', 'snoring_2.0', 'had_menopause_0.0', 'ever_had_miscarriage_stillbirths_or_termination_0.0', 'ever_taken_oral_contraceptive_pill_0.0', 
                   'ever_used_hormone_replacement_therapy_0.0', 'bilateral_oophorectomy_0.0', 'ever_had_hysterectomy_0.0', 'menstruating_today_0.0',
                   'menstruating_today_2.0', 'ever_had_hysterectomy_2.0', 'bilateral_oophorectomy_2.0', 'ever_used_hormone_replacement_therapy_2.0',
                   'af', 'dcm', 'dyspnea', 'hp', 'hf', 'diabetes', 'sf_af', 'sf_hp', 'sf_hf', 'sf_diabetes'
                   'art_stiff_absence_notch_0.0', 'art_stiff_absence_notch_2.0',
                   'spiro_caffeine_last_hour_0.0', 'spiro_inhaler_last_hour_0.0', 'spiro_smoked_last_hour_0.0',
                   'spiro_caffeine_last_hour_2.0', 'spiro_inhaler_last_hour_2.0', 'spiro_smoked_last_hour_2.0'
                   'time_outdoors_summer_0.0', 'time_outdoors_winter_0.0', 'time_outdoors_summer_2.0', 'time_outdoors_winter_2.0',
                   'Chest_pain_0.0', 'Chest_pain_walking_normally_0.0', 'Chest_pain_walking_ceases_standing_still_0.0', 'Chest_pain_walking_uphill_or_hurrying_0.0', 
                   'Chest_pain_2.0', 'Chest_pain_walking_normally_2.0', 'Chest_pain_walking_ceases_standing_still_2.0', 'Chest_pain_walking_uphill_or_hurrying_2.0'
                   'whistling_chest_last_year_0.0', 'shortness_breath_walking_0.0', 'whistling_chest_last_year_2.0', 'shortness_breath_walking_2.0',
                   'leisure_activities_0.0', 'leisure_activities_0.1', 'leisure_activities_0.2', 'leisure_activities_0.3', 'leisure_activities_0.4', 
                   'leisure_activities_2.0', 'leisure_activities_2.1', 'leisure_activities_2.2', 'leisure_activities_2.3', 'leisure_activities_2.4' 
                   'Past_tobacco_smoking_2.0', 'Aclohol_intake_vs_10_prev_years_2.0', 'Former_alcohol_drinker_2.0', 'Ever_Smoked_2.0', 'Current_tobacco_smoking_2.0',
                   'Cooked_vegetable_intake_0.0', 'Salad_raw_vegetable_intake_0.0', 'Processed_meat_intake_0.0', 'Cooked_vegetable_intake_2.0', 
                   'Salad_raw_vegetable_intake_2.0', 'Processed_meat_intake_2.0',
                   'Former_alcohol_drinker_0.0', 'Ever_Smoked_0.0', 'Current_tobacco_smoking_0.0', 'Past_tobacco_smoking_0.0', 'Aclohol_intake_vs_10_prev_years_0.0']


pretty_name_map = {
    "Weight_Img_2.0": "Weight [Image]",
    "bsa_img_2.0": "BSA [Image]",
    "BMI_0.0": "BMI [Baseline]",
    "BMI_2.0": "BMI [Image]",
    "lv_ef_2.0": "LV EF",
    "lv_sv_2.0": "LV SV",
    "lv_esv_2.0": "LV ESV",
    "Weight_0.0": "Weight [Baseline]",
    "ecg_rest_p_axis_2.0": "ECG: P wave axis",
    "spiro_fvc_2.0": "FVC",
    "PulseRate_2.0": "Pulse Rate",
    "prot_albumin_0.0": "Albumin [Baseline]",
    "prot_glucose_0.0": "Glucose [Baseline]",
    "pwa_cardiac_index_2.0": "PWA Cardiac Index",
    "spiro_pef_2.0": "PEF",
    "urine_potassium_0.0": "Potassium in urine [Baseline]",
    "ecg_rest_r_axis_2.0": "ECG: R wave axis",
    "urine_sodium_0.0": "Sodium in urine [Baseline]",
    "pwa_mean_arterial_pressure_2.0": "PWA Mean Arterial Pressure",
    "avg_heart_rate_2.0": "Heart rate",
    "prot_tryglycerides_0.0": "Tryglycerides [Baseline]",
}