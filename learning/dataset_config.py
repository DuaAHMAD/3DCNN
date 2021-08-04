def get_train_val_folders(name):
    if name == "BDI_DERIVED":
        train_exp = [
            
            "Controls/Co_LiBe_NOMS038", "Controls/Co_RoEl_081209",
            "Controls/CoAnBu_090510", "Controls/Co_GlMa_210910", 
            "Controls/Co_MaBa_300910", "Controls/Co_SaMe_020211",
            "Controls/Co_BiLi_71209", "Controls/Co_JeFl_050510",
            "Controls/Co_MiPl_NOMS034", "Controls/Co_StSt_040510",
            "Controls/Co_ChGi_NOMS065", "Controls/Co_JuSm_noms057",
            "Controls/Co_CoPo_050510", "Controls/Co_KaHi_NOMS_040",
            "Controls/Co_PaSo_300410", "Controls/Co_ViHa_NOMS044",
            "Controls/Co_CoTe_189111", "Controls/Co_KaIy_NOMS055",
            "Controls/Co_PrTh_210510", "Controls/Co_DaMe_180510",
            "Controls/Co_LeSm_100310",
            "Patients/Pa_ARI-3248", "Patients/Pa_HSI-2896",
            "Patients/Pa_MBI-3714_NOMS022", "Patients/Pa_MHO_3659_NOMS014",
            "Patients/Pa_HSI_3313", "Patients/Pa_MBI-3725_NOMS028",
            "Patients/Pa_IMC-2911", "Patients/Pa_MHO-3219", 
            "Patients/Pa_PF1-4439_NOMS070", "Patients/Pa_HS1_3529",
            "Patients/Pa_MB1-4149_NOMS050", "Patients/Pa_MHO_3377",
            "Patients/Pa_PF1_3601_noms017", "Patients/Pa_PF_3095",
            "Patients/Pa_HS1-4313_NOMS062", "Patients/Pa_MB1-4307_NOMS061",
            "Patients/Pa_MHO_3441", "Patients/Pa_PF1-3769_noms32",
            "Patients/Pa_PF_3147", "Patients/Pa_HSI-1831",
            "Patients/Pa_MB1-4678_NOMS077",
        ]
        validate_exp = [
            "Controls/Co_AlCh_NOMS076", "Controls/Co_MeYo", 
            "Controls/Co_InGi_NOMS060", "Controls/Co_RoDu_021209",
            "Controls/Co_MoVi_100210", "Controls/Co_VaSi_NOMS069",
            "Patients/Pa_PF1-4213_NOMS058", "Patients/Pa_PF1-3815_NOMS033",
            "Patients/Pa_PF1-4193_NOMS067", "Patients/Pa_PF1_3524",
            "Patients/Pa_MHO3658_noms016Movie", "Patients/Pa_PF13296",
        ]
        
        # Damaged folders: "Controls/Co_EmSp_300310", "Controls/Co_StMc_NOMS063",
        # "Controls/Co_AnNa_NOMS051", "Patients/Pa_StCo", "Patients/Pa_FiDs_GP11",
    elif name == "BDI_DERIVED_SMALL":
        train_exp = [
            "Controls/Co_AlCh_NOMS076", "Controls/CoAnBu_090510", 
            "Controls/Co_BiLi_71209", "Controls/Co_ChGi_NOMS065", 
            "Patients/Pa_ARI-3248", "Patients/Pa_MB1-4678_NOMS077",
            "Patients/Pa_HS1_3529", 
        ]
        validate_exp = [
            "Controls/Co_CoPo_050510", "Patients/Pa_HS1-4313_NOMS062",
            
        ]
    else:
        raise Exception("{} is not a valid train/validation split!".format(name))
    return train_exp, validate_exp