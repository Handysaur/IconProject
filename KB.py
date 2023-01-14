import pytholog as prolog

diseasesKB = prolog.KnowledgeBase("Diseases")

#diseaseHasSymp(symptom,disease)
#doctorStartHour(doctor,start_hour,day)
#doctorEndHour(doctor,end_hour,day)
#doctorAvailability(doctor,hour,day)

diseasesKB(["diseaseHasSymp(itching,Fungal infection)",
"diseaseHasSymp(itching,Fungal infection)",
"diseaseHasSymp(skin_rash,Fungal infection)",
"diseaseHasSymp(nodal_skin_eruptions,Fungal infection)",
"diseaseHasSymp(dischromic,Fungal infection)",
"diseaseHasSymp(_patches,Fungal infection)",

"diseaseHasSymp(continuous_sneezing,Allergy)",
"diseaseHasSymp(shivering,Allergy)",
"diseaseHasSymp(chills,Allergy)",
"diseaseHasSymp(watering_from_eyes,Allergy)",

"diseaseHasSymp(stomach_pain,GERD)",
"diseaseHasSymp(acidity,GERD)",
"diseaseHasSymp(ulcers_on_tongue,GERD)",
"diseaseHasSymp(vomiting,GERD)",
"diseaseHasSymp(cough,GERD)",
"diseaseHasSymp(chest_pain,GERD)",

"diseaseHasSymp(itching,Chronic cholestasis)",
"diseaseHasSymp(vomiting,Chronic cholestasis)",
"diseaseHasSymp(yellowish_skin,Chronic cholestasis)",
"diseaseHasSymp(nausea,Chronic cholestasis)",
"diseaseHasSymp(loss_of_appetite,Chronic cholestasis)",
"diseaseHasSymp(abdominal_pain,Chronic cholestasis)",
"diseaseHasSymp(yellowing_of_eyes,Chronic cholestasis)",

"diseaseHasSymp(itching,Drug Reaction)",
"diseaseHasSymp(skin_rash,Drug Reaction)",
"diseaseHasSymp(stomach_pain,Drug Reaction)",
"diseaseHasSymp(burning_micturition,Drug Reaction)",
"diseaseHasSymp(spotting_,Drug Reaction)",
"diseaseHasSymp(urination,Drug Reaction)",

"diseaseHasSymp(vomiting,Peptic ulcer diseae)",
"diseaseHasSymp(loss_of_appetite,Peptic ulcer diseae)",
"diseaseHasSymp(abdominal_pain,Peptic ulcer diseae)",
"diseaseHasSymp(passage_of_gases,Peptic ulcer diseae)",
"diseaseHasSymp(internal_itching,Peptic ulcer diseae)",

"diseaseHasSymp(muscle_wasting,AIDS)",
"diseaseHasSymp(patches_in_throat,AIDS)",
"diseaseHasSymp(high_fever,AIDS)",
"diseaseHasSymp(extra_marital_contacts,AIDS)",

"diseaseHasSymp(fatigue,Diabetes)",
"diseaseHasSymp(weight_loss,Diabetes)",
"diseaseHasSymp(restlessness,Diabetes)",
"diseaseHasSymp(irregular_sugar_level,Diabetes)",
"diseaseHasSymp(blurred_and_distorted_vision,Diabetes)",
"diseaseHasSymp(obesity,Diabetes)",
"diseaseHasSymp(excessive_hunger,Diabetes)",
"diseaseHasSymp(increased_appetite,Diabetes)",
"diseaseHasSymp(polyuria,Diabetes)",

"diseaseHasSymp(vomiting,Gastroenteritis)",
"diseaseHasSymp(sunken_eyes,Gastroenteritis)",
"diseaseHasSymp(dehydration,Gastroenteritis)",
"diseaseHasSymp(diarrhoea,Gastroenteritis)",

"diseaseHasSymp(fatigue,Bronchial Asthma)",
"diseaseHasSymp(cough,Bronchial Asthma)",
"diseaseHasSymp(high_fever,Bronchial Asthma)",
"diseaseHasSymp(breathlessness,Bronchial Asthma)",
"diseaseHasSymp(family_history,Bronchial Asthma)",
"diseaseHasSymp(mucoid_sputum,Bronchial Asthma)",

"diseaseHasSymp(headache,Hypertension)",
"diseaseHasSymp(chest_pain,Hypertension)",
"diseaseHasSymp(dizziness,Hypertension)",
"diseaseHasSymp(loss_of_balance,Hypertension)",
"diseaseHasSymp(lack_of_concentration,Hypertension)",

"diseaseHasSymp(acidity,Migraine)",
"diseaseHasSymp(indigestion,Migraine)",
"diseaseHasSymp(headache,Migraine)",
"diseaseHasSymp(blurred_and_distorted_vision,Migraine)",
"diseaseHasSymp(excessive_hunger,Migraine)",
"diseaseHasSymp(stiff_neck,Migraine)",
"diseaseHasSymp(depression,Migraine)",
"diseaseHasSymp(irritability,Migraine)",
"diseaseHasSymp(visual_disturbances,Migraine)",

"diseaseHasSymp(back_pain,Cervical spondylosis)",
"diseaseHasSymp(weakness_in_limbs,Cervical spondylosis)",
"diseaseHasSymp(neck_pain,Cervical spondylosis)",
"diseaseHasSymp(dizziness,Cervical spondylosis)",
"diseaseHasSymp(loss_of_balance,Cervical spondylosis)",

"diseaseHasSymp(vomiting,Paralysis)",
"diseaseHasSymp(headache,Paralysis)",
"diseaseHasSymp(weakness_of_one_body_side,Paralysis)",
"diseaseHasSymp(altered_sensorium,Paralysis)",

"diseaseHasSymp(itching,Jaundice)",
"diseaseHasSymp(vomiting,Jaundice)",
"diseaseHasSymp(fatigue,Jaundice)",
"diseaseHasSymp(weight_loss,Jaundice)",
"diseaseHasSymp(high_fever,Jaundice)",
"diseaseHasSymp(yellowish_skin,Jaundice)",
"diseaseHasSymp(dark_urine,Jaundice)",
"diseaseHasSymp(abdominal_pain,Jaundice)",

"diseaseHasSymp(chills,Malaria)",
"diseaseHasSymp(vomiting,Malaria)",
"diseaseHasSymp(high_fever,Malaria)",
"diseaseHasSymp(sweating,Malaria)",
"diseaseHasSymp(headache,Malaria)",
"diseaseHasSymp(nausea,Malaria)",
"diseaseHasSymp(muscle_pain,Malaria)",

"diseaseHasSymp(itching,Chicken pox)",
"diseaseHasSymp(skin_rash,Chicken pox)",
"diseaseHasSymp(fatigue,Chicken pox)",
"diseaseHasSymp(lethargy,Chicken pox)",
"diseaseHasSymp(high_fever,Chicken pox)",
"diseaseHasSymp(headache,Chicken pox)",
"diseaseHasSymp(loss_of_appetite,Chicken pox)",
"diseaseHasSymp(mild_fever,Chicken pox)",
"diseaseHasSymp(swelled_lymph_nodes,Chicken pox)",
"diseaseHasSymp(malaise,Chicken pox)",
"diseaseHasSymp(red_spots_over_body,Chicken pox)",

"diseaseHasSymp(skin_rash,Dengue)",
"diseaseHasSymp(chills,Dengue)",
"diseaseHasSymp(joint_pain,Dengue)",
"diseaseHasSymp(vomiting,Dengue)",
"diseaseHasSymp(fatigue,Dengue)",
"diseaseHasSymp(high_fever,Dengue)",
"diseaseHasSymp(headache,Dengue)",
"diseaseHasSymp(nausea,Dengue)",
"diseaseHasSymp(loss_of_appetite,Dengue)",
"diseaseHasSymp(pain_behind_the_eyes,Dengue)",
"diseaseHasSymp(back_pain,Dengue)",
"diseaseHasSymp(muscle_pain,Dengue)",
"diseaseHasSymp(red_spots_over_body,Dengue)",

"diseaseHasSymp(chills,Typhoid)",
"diseaseHasSymp(vomiting,Typhoid)",
"diseaseHasSymp(fatigue,Typhoid)",
"diseaseHasSymp(high_fever,Typhoid)",
"diseaseHasSymp(nausea,Typhoid)",
"diseaseHasSymp(constipation,Typhoid)",
"diseaseHasSymp(abdominal_pain,Typhoid)",
"diseaseHasSymp(diarrhoea,Typhoid)",
"diseaseHasSymp(toxic_look_,Typhoid)",
"diseaseHasSymp(belly_pain,Typhoid)",

"diseaseHasSymp(joint_pain,hepatitis A)",
"diseaseHasSymp(vomiting,hepatitis A)",
"diseaseHasSymp(yellowish_skin,hepatitis A)",
"diseaseHasSymp(dark_urine,hepatitis A)",
"diseaseHasSymp(nausea,hepatitis A)",
"diseaseHasSymp(loss_of_appetite,hepatitis A)",
"diseaseHasSymp(abdominal_pain,hepatitis A)",
"diseaseHasSymp(diarrhoea,hepatitis A)",
"diseaseHasSymp(mild_fever,hepatitis A)",
"diseaseHasSymp(yellowing_of_eyes,hepatitis A)",
"diseaseHasSymp(muscle_pain,hepatitis A)",

"diseaseHasSymp(itching,Hepatitis B)",
"diseaseHasSymp(fatigue,Hepatitis B)",
"diseaseHasSymp(lethargy,Hepatitis B)",
"diseaseHasSymp(yellowish_skin,Hepatitis B)",
"diseaseHasSymp(dark_urine,Hepatitis B)",
"diseaseHasSymp(loss_of_appetite,Hepatitis B)",
"diseaseHasSymp(abdominal_pain,Hepatitis B)",
"diseaseHasSymp(yellow_urine,Hepatitis B)",
"diseaseHasSymp(yellowing_of_eyes,Hepatitis B)",
"diseaseHasSymp(malaise,Hepatitis B)",
"diseaseHasSymp(receiving_blood_transfusion,Hepatitis B)",
"diseaseHasSymp(receiving_unsterile_injections,Hepatitis B)",

"diseaseHasSymp(fatigue,Hepatitis C)",
"diseaseHasSymp(yellowish_skin,Hepatitis C)",
"diseaseHasSymp(nausea,Hepatitis C)",
"diseaseHasSymp(loss_of_appetite,Hepatitis C)",
"diseaseHasSymp(yellowing_of_eyes,Hepatitis C)",
"diseaseHasSymp(family_history,Hepatitis C)",

"diseaseHasSymp(joint_pain,Hepatitis D)",
"diseaseHasSymp(vomiting,Hepatitis D)",
"diseaseHasSymp(fatigue,Hepatitis D)",
"diseaseHasSymp(yellowish_skin,Hepatitis D)",
"diseaseHasSymp(dark_urine,Hepatitis D)",
"diseaseHasSymp(nausea,Hepatitis D)",
"diseaseHasSymp(loss_of_appetite,Hepatitis D)",
"diseaseHasSymp(abdominal_pain,Hepatitis D)",
"diseaseHasSymp(yellowing_of_eyes,Hepatitis D)",

"diseaseHasSymp(joint_pain,Hepatitis E)",
"diseaseHasSymp(vomiting,Hepatitis E)",
"diseaseHasSymp(fatigue,Hepatitis E)",
"diseaseHasSymp(high_fever,Hepatitis E)",
"diseaseHasSymp(yellowish_skin,Hepatitis E)",
"diseaseHasSymp(dark_urine,Hepatitis E)",
"diseaseHasSymp(nausea,Hepatitis E)",
"diseaseHasSymp(loss_of_appetite,Hepatitis E)",
"diseaseHasSymp(abdominal_pain,Hepatitis E)",
"diseaseHasSymp(yellowing_of_eyes,Hepatitis E)",
"diseaseHasSymp(acute_liver_failure,Hepatitis E)",
"diseaseHasSymp(coma,Hepatitis E)",
"diseaseHasSymp(stomach_bleeding,Hepatitis E)",

"diseaseHasSymp(vomiting,Alcoholic hepatitis)",
"diseaseHasSymp(yellowish_skin,Alcoholic hepatitis)",
"diseaseHasSymp(abdominal_pain,Alcoholic hepatitis)",
"diseaseHasSymp(swelling_of_stomach,Alcoholic hepatitis)",
"diseaseHasSymp(distention_of_abdomen,Alcoholic hepatitis)",
"diseaseHasSymp(history_of_alcohol_consumption,Alcoholic hepatitis)",
"diseaseHasSymp(fluid_overload,Alcoholic hepatitis)",

"diseaseHasSymp(chills,Tuberculosis)",
"diseaseHasSymp(vomiting,Tuberculosis)",
"diseaseHasSymp(fatigue,Tuberculosis)",
"diseaseHasSymp(weight_loss,Tuberculosis)",
"diseaseHasSymp(cough,Tuberculosis)",
"diseaseHasSymp(high_fever,Tuberculosis)",
"diseaseHasSymp(breathlessness,Tuberculosis)",
"diseaseHasSymp(sweating,Tuberculosis)",
"diseaseHasSymp(loss_of_appetite,Tuberculosis)",
"diseaseHasSymp(mild_fever,Tuberculosis)",
"diseaseHasSymp(yellowing_of_eyes,Tuberculosis)",
"diseaseHasSymp(swelled_lymph_nodes,Tuberculosis)",
"diseaseHasSymp(malaise,Tuberculosis)",
"diseaseHasSymp(phlegm,Tuberculosis)",
"diseaseHasSymp(chest_pain,Tuberculosis)",
"diseaseHasSymp(blood_in_sputum,Tuberculosis)",

"diseaseHasSymp(continuous_sneezing,Common Cold)",
"diseaseHasSymp(chills,Common Cold)",
"diseaseHasSymp(fatigue,Common Cold)",
"diseaseHasSymp(cough,Common Cold)",
"diseaseHasSymp(high_fever,Common Cold)",
"diseaseHasSymp(headache,Common Cold)",
"diseaseHasSymp(swelled_lymph_nodes,Common Cold)",
"diseaseHasSymp(malaise,Common Cold)",
"diseaseHasSymp(phlegm,Common Cold)",
"diseaseHasSymp(throat_irritation,Common Cold)",
"diseaseHasSymp(redness_of_eyes,Common Cold)",
"diseaseHasSymp(sinus_pressure,Common Cold)",
"diseaseHasSymp(runny_nose,Common Cold)",
"diseaseHasSymp(congestion,Common Cold)",
"diseaseHasSymp(chest_pain,Common Cold)",
"diseaseHasSymp(loss_of_smell,Common Cold)",
"diseaseHasSymp(muscle_pain,Common Cold)",

"diseaseHasSymp(chills,Pneumonia)",
"diseaseHasSymp(fatigue,Pneumonia)",
"diseaseHasSymp(cough,Pneumonia)",
"diseaseHasSymp(high_fever,Pneumonia)",
"diseaseHasSymp(breathlessness,Pneumonia)",
"diseaseHasSymp(sweating,Pneumonia)",
"diseaseHasSymp(malaise,Pneumonia)",
"diseaseHasSymp(phlegm,Pneumonia)",
"diseaseHasSymp(chest_pain,Pneumonia)",
"diseaseHasSymp(fast_heart_rate,Pneumonia)",
"diseaseHasSymp(rusty_sputum,Pneumonia)",

"diseaseHasSymp(constipation,Dimorphic hemmorhoids)",
"diseaseHasSymp(pain_during_bowel_movements,Dimorphic hemmorhoids)",
"diseaseHasSymp(pain_in_anal_region,Dimorphic hemmorhoids)",
"diseaseHasSymp(bloody_stool,Dimorphic hemmorhoids)",
"diseaseHasSymp(irritation_in_anus,Dimorphic hemmorhoids)",

"diseaseHasSymp(vomiting,Heart attack)",
"diseaseHasSymp(breathlessness,Heart attack)",
"diseaseHasSymp(sweating,Heart attack)",
"diseaseHasSymp(chest_pain,Heart attack)",

"diseaseHasSymp(fatigue,Varicose veins)",
"diseaseHasSymp(cramps,Varicose veins)",
"diseaseHasSymp(bruising,Varicose veins)",
"diseaseHasSymp(obesity,Varicose veins)",
"diseaseHasSymp(swollen_legs,Varicose veins)",
"diseaseHasSymp(swollen_blood_vessels,Varicose veins)",
"diseaseHasSymp(prominent_veins_on_calf,Varicose veins)",

"diseaseHasSymp(fatigue,Hypothyroidism)",
"diseaseHasSymp(weight_gain,Hypothyroidism)",
"diseaseHasSymp(cold_hands_and_feets,Hypothyroidism)",
"diseaseHasSymp(mood_swings,Hypothyroidism)",
"diseaseHasSymp(lethargy,Hypothyroidism)",
"diseaseHasSymp(dizziness,Hypothyroidism)",
"diseaseHasSymp(puffy_face_and_eyes,Hypothyroidism)",
"diseaseHasSymp(enlarged_thyroid,Hypothyroidism)",
"diseaseHasSymp(brittle_nails,Hypothyroidism)",
"diseaseHasSymp(swollen_extremeties,Hypothyroidism)",
"diseaseHasSymp(depression,Hypothyroidism)",
"diseaseHasSymp(irritability,Hypothyroidism)",
"diseaseHasSymp(abnormal_menstruation,Hypothyroidism)",

"diseaseHasSymp(fatigue,Hyperthyroidism)",
"diseaseHasSymp(mood_swings,Hyperthyroidism)",
"diseaseHasSymp(weight_loss,Hyperthyroidism)",
"diseaseHasSymp(restlessness,Hyperthyroidism)",
"diseaseHasSymp(sweating,Hyperthyroidism)",
"diseaseHasSymp(diarrhoea,Hyperthyroidism)",
"diseaseHasSymp(fast_heart_rate,Hyperthyroidism)",
"diseaseHasSymp(excessive_hunger,Hyperthyroidism)",
"diseaseHasSymp(muscle_weakness,Hyperthyroidism)",
"diseaseHasSymp(irritability,Hyperthyroidism)",
"diseaseHasSymp(abnormal_menstruation,Hyperthyroidism)",

"diseaseHasSymp(vomiting,Hypoglycemia)",
"diseaseHasSymp(fatigue,Hypoglycemia)",
"diseaseHasSymp(anxiety,Hypoglycemia)",
"diseaseHasSymp(sweating,Hypoglycemia)",
"diseaseHasSymp(headache,Hypoglycemia)",
"diseaseHasSymp(nausea,Hypoglycemia)",
"diseaseHasSymp(blurred_and_distorted_vision,Hypoglycemia)",
"diseaseHasSymp(excessive_hunger,Hypoglycemia)",
"diseaseHasSymp(drying_and_tingling_lips,Hypoglycemia)",
"diseaseHasSymp(slurred_speech,Hypoglycemia)",
"diseaseHasSymp(irritability,Hypoglycemia)",
"diseaseHasSymp(palpitations,Hypoglycemia)",

"diseaseHasSymp(joint_pain,Osteoarthristis)",
"diseaseHasSymp(neck_pain,Osteoarthristis)",
"diseaseHasSymp(knee_pain,Osteoarthristis)",
"diseaseHasSymp(hip_joint_pain,Osteoarthristis)",
"diseaseHasSymp(swelling_joints,Osteoarthristis)",
"diseaseHasSymp(painful_walking,Osteoarthristis)",

"diseaseHasSymp(muscle_weakness,Arthritis)",
"diseaseHasSymp(stiff_neck,Arthritis)",
"diseaseHasSymp(swelling_joints,Arthritis)",
"diseaseHasSymp(movement_stiffness,Arthritis)",
"diseaseHasSymp(painful_walking,Arthritis)",

"diseaseHasSymp(vomiting,Paroymsal  Positional Vertigo)",
"diseaseHasSymp(headache,Paroymsal  Positional Vertigo)",
"diseaseHasSymp(nausea,Paroymsal  Positional Vertigo)",
"diseaseHasSymp(spinning_movements,Paroymsal  Positional Vertigo)",
"diseaseHasSymp(loss_of_balance,Paroymsal  Positional Vertigo)",
"diseaseHasSymp(unsteadiness,Paroymsal  Positional Vertigo)",

"diseaseHasSymp(skin_rash,Acne)",
"diseaseHasSymp(pus_filled_pimples,Acne)",
"diseaseHasSymp(blackheads,Acne)",
"diseaseHasSymp(scurring,Acne)",

"diseaseHasSymp(burning_micturition,Urinary tract infection)",
"diseaseHasSymp(bladder_discomfort,Urinary tract infection)",
"diseaseHasSymp(foul_smell_of_urine,Urinary tract infection)",
"diseaseHasSymp(continuous_feel_of_urine,Urinary tract infection)",

"diseaseHasSymp(skin_rash,Psoriasis)",
"diseaseHasSymp(joint_pain,Psoriasis)",
"diseaseHasSymp(skin_peeling,Psoriasis)",
"diseaseHasSymp(silver_like_dusting,Psoriasis)",
"diseaseHasSymp(small_dents_in_nails,Psoriasis)",
"diseaseHasSymp(inflammatory_nails,Psoriasis)",

"diseaseHasSymp(skin_rash,Impetigo)",
"diseaseHasSymp(high_fever,Impetigo)",
"diseaseHasSymp(blister,Impetigo)",
"diseaseHasSymp(red_sore_around_nose,Impetigo)",
"diseaseHasSymp(yellow_crust_ooze,Impetigo)",

"doctorStartHour(Scalera,10,Lunedì)",
"doctorStartHour(Scalera,09,Mercoledì)",
"doctorStartHour(Scalera,11,Giovedì)",

"doctorStartHour(Raguso,11,Lunedì)",
"doctorStartHour(Raguso,09,Giovedì)",
"doctorStartHour(Raguso,15,Venerdì)",

"doctorStartHour(Natuzzi-Caggiano,09,Lunedì)",
"doctorStartHour(Natuzzi-Caggiano,09,Martedì)",
"doctorStartHour(Natuzzi-Caggiano,15,Giovedì)",
"doctorStartHour(Natuzzi-Caggiano,10,Sabato)",

"doctorStartHour(Capurso,10,Giovedì)",

"doctorStartHour(Incampo,16,Lunedì)",
"doctorStartHour(Incampo,09,Mercoledì)",
"doctorStartHour(Incampo,17,Giovedì)",
"doctorStartHour(Incampo,11,Venerdì)",

"doctorStartHour(Colasuonno,09,Mercoledì)",
"doctorStartHour(Colasuonno,12,Giovedì)",
"doctorStartHour(Colasuonno,11,Venerdì)",

"doctorStartHour(Farella,11,Lunedì)",
"doctorStartHour(Farella,12,Giovedì)",
"doctorStartHour(Farella,11,Sabato)",

"doctorEndHour(Farella,18,Lunedì)",
"doctorEndHour(Farella,17,Giovedì)",
"doctorEndHour(Farella,15,Sabato)",

"doctorEndHour(Colasuonno,13,Mercoledì)",
"doctorEndHour(Colasuonno,17,Giovedì)",
"doctorEndHour(Colasuonno,20,Venerdì)",

"doctorEndHour(Incampo,20,Lunedì)",
"doctorEndHour(Incampo,12,Mercoledì)",
"doctorEndHour(Incampo,20,Giovedì)",
"doctorEndHour(Incampo,13,Venerdì)",

"doctorEndHour(Capurso,19,Giovedì)",

"doctorEndHour(Natuzzi-Caggiano,13,Lunedì)",
"doctorEndHour(Natuzzi-Caggiano,12,Martedì)",
"doctorEndHour(Natuzzi-Caggiano,20,Giovedì)",
"doctorStartHour(Natuzzi-Caggiano,13,Sabato)",

"doctorStartHour(Denora,12,Lunedì)",
"doctorStartHour(Denora,10,Martedì)",

"doctorEndHour(Denora,18,Lunedì)",
"doctorEndHour(Denora,16,Martedì)",

"doctorEndHour(Raguso,13,Lunedì)",
"doctorEndHour(Raguso,12,Giovedì)",
"doctorEndHour(Raguso,20,Venerdì)",

"doctorEndHour(Scalera,13,Lunedì)",
"doctorEndHour(Scalera,12,Mercoledì)",
"doctorEndHour(Scalera,13,Giovedì)",

"doctorAvailability(doctor,hour,day):-doctorEndHour(doctor,start_hour,day),doctorEndHour(doctor,end_hour,day),hour < end_hour, hour > start_hour",
])


"""
Metodo per effettuare una query alla knowledge base
Prende in input la knowledge base da interrogare e una stringa contenente la query
Restituisce l'insieme dei risultati
"""
def askANDResolve_KB(kb:prolog.KnowledgeBase, askStr):
    kb_Results = set()
    query = kb.query(prolog.Expr(askStr))

    for result in query:
        kb_Results.add(str(result))

    return kb_Results    