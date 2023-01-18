import pytholog as prolog

diseasesKB = prolog.KnowledgeBase("Diseases")

#diseasehassymp(symptom,disease)
#doctorstarthour(doctor,start_hour,day)
#doctorendhour(doctor,end_hour,day)

diseasesKB(["diseasehassymp(itching,fungal_infection)",
            "diseasehassymp(itching,fungal_infection)",
            "diseasehassymp(skin_rash,fungal_infection)",
            "diseasehassymp(nodal_skin_eruptions,fungal_infection)",
            "diseasehassymp(dischromic,fungal_infection)",
            "diseasehassymp(patches,fungal_infection)",

            "diseasehassymp(continuous_sneezing,allergy)",
            "diseasehassymp(shivering,allergy)",
            "diseasehassymp(chills,allergy)",
            "diseasehassymp(watering_from_eyes,allergy)",

            "diseasehassymp(stomach_pain,gerd)",
            "diseasehassymp(acidity,gerd)",
            "diseasehassymp(ulcers_on_tongue,gerd)",
            "diseasehassymp(vomiting,gerd)",
            "diseasehassymp(cough,gerd)",
            "diseasehassymp(chest_pain,gerd)",

            "diseasehassymp(itching,chronic_cholestasis)",
            "diseasehassymp(vomiting,chronic_cholestasis)",
            "diseasehassymp(yellowish_skin,chronic_cholestasis)",
            "diseasehassymp(nausea,chronic_cholestasis)",
            "diseasehassymp(loss_of_appetite,chronic_cholestasis)",
            "diseasehassymp(abdominal_pain,chronic_cholestasis)",
            "diseasehassymp(yellowing_of_eyes,chronic_cholestasis)",

            "diseasehassymp(itching,drug_reaction)",
            "diseasehassymp(skin_rash,drug_reaction)",
            "diseasehassymp(stomach_pain,drug_reaction)",
            "diseasehassymp(burning_micturition,drug_reaction)",
            "diseasehassymp(spotting_,drug_reaction)",
            "diseasehassymp(urination,drug_reaction)",

            "diseasehassymp(vomiting,peptic_ulcer_diseae)",
            "diseasehassymp(loss_of_appetite,peptic_ulcer_diseae)",
            "diseasehassymp(abdominal_pain,peptic_ulcer_diseae)",
            "diseasehassymp(passage_of_gases,peptic_ulcer_diseae)",
            "diseasehassymp(internal_itching,peptic_ulcer_diseae)",

            "diseasehassymp(muscle_wasting,aids)",
            "diseasehassymp(patches_in_throat,aids)",
            "diseasehassymp(high_fever,aids)",
            "diseasehassymp(extra_marital_contacts,aids)",

            "diseasehassymp(fatigue,diabetes)",
            "diseasehassymp(weight_loss,diabetes)",
            "diseasehassymp(restlessness,diabetes)",
            "diseasehassymp(irregular_sugar_level,diabetes)",
            "diseasehassymp(blurred_and_distorted_vision,diabetes)",
            "diseasehassymp(obesity,diabetes)",
            "diseasehassymp(excessive_hunger,diabetes)",
            "diseasehassymp(increased_appetite,diabetes)",
            "diseasehassymp(polyuria,diabetes)",

            "diseasehassymp(vomiting,gastroenteritis)",
            "diseasehassymp(sunken_eyes,gastroenteritis)",
            "diseasehassymp(dehydration,gastroenteritis)",
            "diseasehassymp(diarrhoea,gastroenteritis)",

            "diseasehassymp(fatigue,bronchial_asthma)",
            "diseasehassymp(cough,bronchial_asthma)",
            "diseasehassymp(high_fever,bronchial_asthma)",
            "diseasehassymp(breathlessness,bronchial_asthma)",
            "diseasehassymp(family_history,bronchial_asthma)",
            "diseasehassymp(mucoid_sputum,bronchial_asthma)",

            "diseasehassymp(headache,hypertension)",
            "diseasehassymp(chest_pain,hypertension)",
            "diseasehassymp(dizziness,hypertension)",
            "diseasehassymp(loss_of_balance,hypertension)",
            "diseasehassymp(lack_of_concentration,hypertension)",

            "diseasehassymp(acidity,migraine)",
            "diseasehassymp(indigestion,migraine)",
            "diseasehassymp(headache,migraine)",
            "diseasehassymp(blurred_and_distorted_vision,migraine)",
            "diseasehassymp(excessive_hunger,migraine)",
            "diseasehassymp(stiff_neck,migraine)",
            "diseasehassymp(depression,migraine)",
            "diseasehassymp(irritability,migraine)",
            "diseasehassymp(visual_disturbances,migraine)",

            "diseasehassymp(back_pain,cervical_spondylosis)",
            "diseasehassymp(weakness_in_limbs,cervical_spondylosis)",
            "diseasehassymp(neck_pain,cervical_spondylosis)",
            "diseasehassymp(dizziness,cervical_spondylosis)",
            "diseasehassymp(loss_of_balance,cervical_spondylosis)",

            "diseasehassymp(vomiting,paralysis)",
            "diseasehassymp(headache,paralysis)",
            "diseasehassymp(weakness_of_one_body_side,paralysis)",
            "diseasehassymp(altered_sensorium,paralysis)",

            "diseasehassymp(itching,jaundice)",
            "diseasehassymp(vomiting,jaundice)",
            "diseasehassymp(fatigue,jaundice)",
            "diseasehassymp(weight_loss,jaundice)",
            "diseasehassymp(high_fever,jaundice)",
            "diseasehassymp(yellowish_skin,jaundice)",
            "diseasehassymp(dark_urine,jaundice)",
            "diseasehassymp(abdominal_pain,jaundice)",

            "diseasehassymp(chills,malaria)",
            "diseasehassymp(vomiting,malaria)",
            "diseasehassymp(high_fever,malaria)",
            "diseasehassymp(sweating,malaria)",
            "diseasehassymp(headache,malaria)",
            "diseasehassymp(nausea,malaria)",
            "diseasehassymp(muscle_pain,malaria)",

            "diseasehassymp(itching,chicken_pox)",
            "diseasehassymp(skin_rash,chicken_pox)",
            "diseasehassymp(fatigue,chicken_pox)",
            "diseasehassymp(lethargy,chicken_pox)",
            "diseasehassymp(high_fever,chicken_pox)",
            "diseasehassymp(headache,chicken_pox)",
            "diseasehassymp(loss_of_appetite,chicken_pox)",
            "diseasehassymp(mild_fever,chicken_pox)",
            "diseasehassymp(swelled_lymph_nodes,chicken_pox)",
            "diseasehassymp(malaise,chicken_pox)",
            "diseasehassymp(red_spots_over_body,chicken_pox)",

            "diseasehassymp(skin_rash,dengue)",
            "diseasehassymp(chills,dengue)",
            "diseasehassymp(joint_pain,dengue)",
            "diseasehassymp(vomiting,dengue)",
            "diseasehassymp(fatigue,dengue)",
            "diseasehassymp(high_fever,dengue)",
            "diseasehassymp(headache,dengue)",
            "diseasehassymp(nausea,dengue)",
            "diseasehassymp(loss_of_appetite,dengue)",
            "diseasehassymp(pain_behind_the_eyes,dengue)",
            "diseasehassymp(back_pain,dengue)",
            "diseasehassymp(muscle_pain,dengue)",
            "diseasehassymp(red_spots_over_body,dengue)",

            "diseasehassymp(chills,typhoid)",
            "diseasehassymp(vomiting,typhoid)",
            "diseasehassymp(fatigue,typhoid)",
            "diseasehassymp(high_fever,typhoid)",
            "diseasehassymp(nausea,typhoid)",
            "diseasehassymp(constipation,typhoid)",
            "diseasehassymp(abdominal_pain,typhoid)",
            "diseasehassymp(diarrhoea,typhoid)",
            "diseasehassymp(toxic_look_,typhoid)",
            "diseasehassymp(belly_pain,typhoid)",

            "diseasehassymp(joint_pain,hepatitis_a)",
            "diseasehassymp(vomiting,hepatitis_a)",
            "diseasehassymp(yellowish_skin,hepatitis_a)",
            "diseasehassymp(dark_urine,hepatitis_a)",
            "diseasehassymp(nausea,hepatitis_a)",
            "diseasehassymp(loss_of_appetite,hepatitis_a)",
            "diseasehassymp(abdominal_pain,hepatitis_a)",
            "diseasehassymp(diarrhoea,hepatitis_a)",
            "diseasehassymp(mild_fever,hepatitis_a)",
            "diseasehassymp(yellowing_of_eyes,hepatitis_a)",
            "diseasehassymp(muscle_pain,hepatitis_a)",

            "diseasehassymp(itching,hepatitis_b)",
            "diseasehassymp(fatigue,hepatitis_b)",
            "diseasehassymp(lethargy,hepatitis_b)",
            "diseasehassymp(yellowish_skin,hepatitis_b)",
            "diseasehassymp(dark_urine,hepatitis_b)",
            "diseasehassymp(loss_of_appetite,hepatitis_b)",
            "diseasehassymp(abdominal_pain,hepatitis_b)",
            "diseasehassymp(yellow_urine,hepatitis_b)",
            "diseasehassymp(yellowing_of_eyes,hepatitis_b)",
            "diseasehassymp(malaise,hepatitis_b)",
            "diseasehassymp(receiving_blood_transfusion,hepatitis_b)",
            "diseasehassymp(receiving_unsterile_injections,hepatitis_b)",

            "diseasehassymp(fatigue,hepatitis_c)",
            "diseasehassymp(yellowish_skin,hepatitis_c)",
            "diseasehassymp(nausea,hepatitis_c)",
            "diseasehassymp(loss_of_appetite,hepatitis_c)",
            "diseasehassymp(yellowing_of_eyes,hepatitis_c)",
            "diseasehassymp(family_history,hepatitis_c)",

            "diseasehassymp(joint_pain,hepatitis_d)",
            "diseasehassymp(vomiting,hepatitis_d)",
            "diseasehassymp(fatigue,hepatitis_d)",
            "diseasehassymp(yellowish_skin,hepatitis_d)",
            "diseasehassymp(dark_urine,hepatitis_d)",
            "diseasehassymp(nausea,hepatitis_d)",
            "diseasehassymp(loss_of_appetite,hepatitis_d)",
            "diseasehassymp(abdominal_pain,hepatitis_d)",
            "diseasehassymp(yellowing_of_eyes,hepatitis_d)",

            "diseasehassymp(joint_pain,hepatitis_e)",
            "diseasehassymp(vomiting,hepatitis_e)",
            "diseasehassymp(fatigue,hepatitis_e)",
            "diseasehassymp(high_fever,hepatitis_e)",
            "diseasehassymp(yellowish_skin,hepatitis_e)",
            "diseasehassymp(dark_urine,hepatitis_e)",
            "diseasehassymp(nausea,hepatitis_e)",
            "diseasehassymp(loss_of_appetite,hepatitis_e)",
            "diseasehassymp(abdominal_pain,hepatitis_e)",
            "diseasehassymp(yellowing_of_eyes,hepatitis_e)",
            "diseasehassymp(acute_liver_failure,hepatitis_e)",
            "diseasehassymp(coma,hepatitis_e)",
            "diseasehassymp(stomach_bleeding,hepatitis_e)",

            "diseasehassymp(vomiting,alcoholic_hepatitis)",
            "diseasehassymp(yellowish_skin,alcoholic_hepatitis)",
            "diseasehassymp(abdominal_pain,alcoholic_hepatitis)",
            "diseasehassymp(swelling_of_stomach,alcoholic_hepatitis)",
            "diseasehassymp(distention_of_abdomen,alcoholic_hepatitis)",
            "diseasehassymp(history_of_alcohol_consumption,alcoholic_hepatitis)",
            "diseasehassymp(fluid_overload,alcoholic_hepatitis)",

            "diseasehassymp(chills,tuberculosis)",
            "diseasehassymp(vomiting,tuberculosis)",
            "diseasehassymp(fatigue,tuberculosis)",
            "diseasehassymp(weight_loss,tuberculosis)",
            "diseasehassymp(cough,tuberculosis)",
            "diseasehassymp(high_fever,tuberculosis)",
            "diseasehassymp(breathlessness,tuberculosis)",
            "diseasehassymp(sweating,tuberculosis)",
            "diseasehassymp(loss_of_appetite,tuberculosis)",
            "diseasehassymp(mild_fever,tuberculosis)",
            "diseasehassymp(yellowing_of_eyes,tuberculosis)",
            "diseasehassymp(swelled_lymph_nodes,tuberculosis)",
            "diseasehassymp(malaise,tuberculosis)",
            "diseasehassymp(phlegm,tuberculosis)",
            "diseasehassymp(chest_pain,tuberculosis)",
            "diseasehassymp(blood_in_sputum,tuberculosis)",

            "diseasehassymp(continuous_sneezing,common_cold)",
            "diseasehassymp(chills,common_cold)",
            "diseasehassymp(fatigue,common_cold)",
            "diseasehassymp(cough,common_cold)",
            "diseasehassymp(high_fever,common_cold)",
            "diseasehassymp(headache,common_cold)",
            "diseasehassymp(swelled_lymph_nodes,common_cold)",
            "diseasehassymp(malaise,common_cold)",
            "diseasehassymp(phlegm,common_cold)",
            "diseasehassymp(throat_irritation,common_cold)",
            "diseasehassymp(redness_of_eyes,common_cold)",
            "diseasehassymp(sinus_pressure,common_cold)",
            "diseasehassymp(runny_nose,common_cold)",
            "diseasehassymp(congestion,common_cold)",
            "diseasehassymp(chest_pain,common_cold)",
            "diseasehassymp(loss_of_smell,common_cold)",
            "diseasehassymp(muscle_pain,common_cold)",

            "diseasehassymp(chills,pneumonia)",
            "diseasehassymp(fatigue,pneumonia)",
            "diseasehassymp(cough,pneumonia)",
            "diseasehassymp(high_fever,pneumonia)",
            "diseasehassymp(breathlessness,pneumonia)",
            "diseasehassymp(sweating,pneumonia)",
            "diseasehassymp(malaise,pneumonia)",
            "diseasehassymp(phlegm,pneumonia)",
            "diseasehassymp(chest_pain,pneumonia)",
            "diseasehassymp(fast_heart_rate,pneumonia)",
            "diseasehassymp(rusty_sputum,pneumonia)",

            "diseasehassymp(constipation,dimorphic_hemmorhoids)",
            "diseasehassymp(pain_during_bowel_movements,dimorphic_hemmorhoids)",
            "diseasehassymp(pain_in_anal_region,dimorphic_hemmorhoids)",
            "diseasehassymp(bloody_stool,dimorphic_hemmorhoids)",
            "diseasehassymp(irritation_in_anus,dimorphic_hemmorhoids)",

            "diseasehassymp(vomiting,heart_attack)",
            "diseasehassymp(breathlessness,heart_attack)",
            "diseasehassymp(sweating,heart_attack)",
            "diseasehassymp(chest_pain,heart_attack)",

            "diseasehassymp(fatigue,varicose_veins)",
            "diseasehassymp(cramps,varicose_veins)",
            "diseasehassymp(bruising,varicose_veins)",
            "diseasehassymp(obesity,varicose_veins)",
            "diseasehassymp(swollen_legs,varicose_veins)",
            "diseasehassymp(swollen_blood_vessels,varicose_veins)",
            "diseasehassymp(prominent_veins_on_calf,varicose_veins)",

            "diseasehassymp(fatigue,hypothyroidism)",
            "diseasehassymp(weight_gain,hypothyroidism)",
            "diseasehassymp(cold_hands_and_feets,hypothyroidism)",
            "diseasehassymp(mood_swings,hypothyroidism)",
            "diseasehassymp(lethargy,hypothyroidism)",
            "diseasehassymp(dizziness,hypothyroidism)",
            "diseasehassymp(puffy_face_and_eyes,hypothyroidism)",
            "diseasehassymp(enlarged_thyroid,hypothyroidism)",
            "diseasehassymp(brittle_nails,hypothyroidism)",
            "diseasehassymp(swollen_extremeties,hypothyroidism)",
            "diseasehassymp(depression,hypothyroidism)",
            "diseasehassymp(irritability,hypothyroidism)",
            "diseasehassymp(abnormal_menstruation,hypothyroidism)",

            "diseasehassymp(fatigue,hyperthyroidism)",
            "diseasehassymp(mood_swings,hyperthyroidism)",
            "diseasehassymp(weight_loss,hyperthyroidism)",
            "diseasehassymp(restlessness,hyperthyroidism)",
            "diseasehassymp(sweating,hyperthyroidism)",
            "diseasehassymp(diarrhoea,hyperthyroidism)",
            "diseasehassymp(fast_heart_rate,hyperthyroidism)",
            "diseasehassymp(excessive_hunger,hyperthyroidism)",
            "diseasehassymp(muscle_weakness,hyperthyroidism)",
            "diseasehassymp(irritability,hyperthyroidism)",
            "diseasehassymp(abnormal_menstruation,hyperthyroidism)",

            "diseasehassymp(vomiting,hypoglycemia)",
            "diseasehassymp(fatigue,hypoglycemia)",
            "diseasehassymp(anxiety,hypoglycemia)",
            "diseasehassymp(sweating,hypoglycemia)",
            "diseasehassymp(headache,hypoglycemia)",
            "diseasehassymp(nausea,hypoglycemia)",
            "diseasehassymp(blurred_and_distorted_vision,hypoglycemia)",
            "diseasehassymp(excessive_hunger,hypoglycemia)",
            "diseasehassymp(drying_and_tingling_lips,hypoglycemia)",
            "diseasehassymp(slurred_speech,hypoglycemia)",
            "diseasehassymp(irritability,hypoglycemia)",
            "diseasehassymp(palpitations,hypoglycemia)",

            "diseasehassymp(joint_pain,osteoarthristis)",
            "diseasehassymp(neck_pain,osteoarthristis)",
            "diseasehassymp(knee_pain,osteoarthristis)",
            "diseasehassymp(hip_joint_pain,osteoarthristis)",
            "diseasehassymp(swelling_joints,osteoarthristis)",
            "diseasehassymp(painful_walking,osteoarthristis)",

            "diseasehassymp(muscle_weakness,arthritis)",
            "diseasehassymp(stiff_neck,arthritis)",
            "diseasehassymp(swelling_joints,arthritis)",
            "diseasehassymp(movement_stiffness,arthritis)",
            "diseasehassymp(painful_walking,arthritis)",

            "diseasehassymp(vomiting,paroymsal__positional_vertigo)",
            "diseasehassymp(headache,paroymsal__positional_vertigo)",
            "diseasehassymp(nausea,paroymsal__positional_vertigo)",
            "diseasehassymp(spinning_movements,paroymsal__positional_vertigo)",
            "diseasehassymp(loss_of_balance,paroymsal__positional_vertigo)",
            "diseasehassymp(unsteadiness,paroymsal__positional_vertigo)",

            "diseasehassymp(skin_rash,acne)",
            "diseasehassymp(pus_filled_pimples,acne)",
            "diseasehassymp(blackheads,acne)",
            "diseasehassymp(scurring,acne)",

            "diseasehassymp(burning_micturition,urinary_tract_infection)",
            "diseasehassymp(bladder_discomfort,urinary_tract_infection)",
            "diseasehassymp(foul_smell_of_urine,urinary_tract_infection)",
            "diseasehassymp(continuous_feel_of_urine,urinary_tract_infection)",

            "diseasehassymp(skin_rash,psoriasis)",
            "diseasehassymp(joint_pain,psoriasis)",
            "diseasehassymp(skin_peeling,psoriasis)",
            "diseasehassymp(silver_like_dusting,psoriasis)",
            "diseasehassymp(small_dents_in_nails,psoriasis)",
            "diseasehassymp(inflammatory_nails,psoriasis)",

            "diseasehassymp(skin_rash,impetigo)",
            "diseasehassymp(high_fever,impetigo)",
            "diseasehassymp(blister,impetigo)",
            "diseasehassymp(red_sore_around_nose,impetigo)",
            "diseasehassymp(yellow_crust_ooze,impetigo)",

            "disease(X):-diseasehassymp(Y,X),write(Y),nl",

            "doctorstarthour(scalera,10,lunedi)",
            "doctorstarthour(scalera,9,mercoledi)",
            "doctorstarthour(scalera,11,giovedi)",
            "doctorendhour(scalera,13,lunedi)",
            "doctorendhour(scalera,12,mercoledi)",
            "doctorendhour(scalera,13,giovedi)",            

            "doctorstarthour(raguso,11,lunedi)",
            "doctorstarthour(raguso,9,giovedi)",
            "doctorstarthour(raguso,15,venerdi)",
            "doctorendhour(raguso,13,lunedi)",
            "doctorendhour(raguso,12,giovedi)",
            "doctorendhour(raguso,20,venerdi)",            

            "doctorstarthour(natuzzi-caggiano,9,lunedi)",
            "doctorstarthour(natuzzi-caggiano,9,martedi)",
            "doctorstarthour(natuzzi-caggiano,15,giovedi)",
            "doctorstarthour(natuzzi-caggiano,10,sabato)",
            "doctorendhour(natuzzi-caggiano,13,lunedi)",
            "doctorendhour(natuzzi-caggiano,12,martedi)",
            "doctorendhour(natuzzi-caggiano,20,giovedi)",
            "doctorendhour(natuzzi-caggiano,13,sabato)",            

            "doctorstarthour(capurso,10,giovedi)",
            "doctorendhour(capurso,19,giovedi)",            

            "doctorstarthour(incampo,16,lunedi)",
            "doctorstarthour(incampo,9,mercoledi)",
            "doctorstarthour(incampo,17,giovedi)",
            "doctorstarthour(incampo,11,venerdi)",
            "doctorendhour(incampo,20,lunedi)",
            "doctorendhour(incampo,12,mercoledi)",
            "doctorendhour(incampo,20,giovedi)",
            "doctorendhour(incampo,13,venerdi)",            

            "doctorstarthour(colasuonno,9,mercoledi)",
            "doctorstarthour(colasuonno,12,giovedi)",
            "doctorstarthour(colasuonno,11,venerdi)",
            "doctorendhour(colasuonno,13,mercoledi)",
            "doctorendhour(colasuonno,17,giovedi)",
            "doctorendhour(colasuonno,20,venerdi)",

            "doctorstarthour(farella,11,lunedi)",
            "doctorstarthour(farella,12,giovedi)",
            "doctorstarthour(farella,11,sabato)",
            "doctorendhour(farella,18,lunedi)",
            "doctorendhour(farella,17,giovedi)",
            "doctorendhour(farella,15,sabato)",

            "doctorstarthour(denora,12,lunedi)",
            "doctorstarthour(denora,10,martedi)",
            "doctorendhour(denora,18,lunedi)",
            "doctorendhour(denora,16,martedi)"
])

"""
Metodo_per_effettuare_una_query_alla_knowledge_base
Prende_in_input_la_knowledge_base_da_interrogare_e_una_stringa_contenente_la_query
Restituisce_l'insieme_dei_risultati
"""
def askANDResolve_KB(kb:prolog.KnowledgeBase, askStr):
    kb_Results = set()
    kb_Results_Values = []
    query = kb.query(prolog.Expr(askStr))
    print(askStr)
    for result in query:
        print(result)
        kb_Results.add(str(result))
        kb_Results_Values.append(result.get('X'))

    return kb_Results,kb_Results_Values 