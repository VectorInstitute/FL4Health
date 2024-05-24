import os, json
import pandas as pd

if __name__ == "__main__" :

    data_path = "fl4health/utils/datasets"

    # Load GT file of ISIC-19
    ISIC_2019_path = f"{data_path}/ISIC_2019"
    ISIC_csv_path = os.path.join( ISIC_2019_path, "ISIC_2019_Training_GroundTruth.csv")
    ISIC_df = pd.read_csv(ISIC_csv_path)

    # Load metadata file of HAM10000
    HAM_10000_path = f"{data_path}/HAM10000"
    HAM_csv_path = os.path.join( HAM_10000_path, "HAM10000_metadata")
    HAM_df = pd.read_csv(HAM_csv_path)

    # Delete duplicate images in ISIC-19
    ISIC_meta = pd.read_csv(os.path.join( ISIC_2019_path, "ISIC_2019_Training_Metadata.csv"))
    barcelona_list = [ i for i in ISIC_meta['lesion_id'].dropna() if 'BCN' in i ]
    barcelona_core = ISIC_meta[ISIC_meta['lesion_id'].isin(barcelona_list)]

    core_2019 = ISIC_df[ISIC_df['image'].isin(barcelona_core['image'])]
    core_2019.to_csv(os.path.join( ISIC_2019_path, "ISIC_2019_core.csv") , mode='w')

    # Split ronsendahl and vienna in HAM10000
    rosendahl_data = HAM_df[ HAM_df['dataset'] == 'rosendahl' ]
    rosendahl_data.to_csv(os.path.join( HAM_10000_path, "HAM_ronsendahl.csv") , mode='w')
    vienna_data = HAM_df[ HAM_df['dataset'] != 'rosendahl']
    vienna_data.to_csv(os.path.join( HAM_10000_path, "HAM_vienna.csv") , mode='w')

    # Load PAD-UFES-20
    PAD_UFES_20_path = f"{data_path}/PAD-UFES-20"
    PAD_UFES_20_csv_path = os.path.join( PAD_UFES_20_path, "metadata.csv")
    PAD_UFES_20 = pd.read_csv(PAD_UFES_20_csv_path)

    # Load Derm7pt
    Derm7pt_path = f"{data_path}/Derm7pt"
    Derm7pt_csv_path = os.path.join( Derm7pt_path, "meta/meta.csv" )
    Derm7pt = pd.read_csv(Derm7pt_csv_path)

    derm7pt_labelmap = {
    'lentigo' : 'MISC',
    'melanosis' : 'MISC',
    'miscellaneous' : 'MISC',
    }

    MISC_list = ['lentigo', 'melanosis', 'miscellaneous']
    Derm7pt_core = Derm7pt[~Derm7pt['diagnosis'].isin( MISC_list )]
    Derm7pt_core.to_csv( os.path.join( Derm7pt_path, "meta/meta_core.csv" ))

    ####################################################################################################################################
    ## Unify the naming of labels in the below section

    official_columns = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

    # Client : Barcelona
    ############################################################################################################################
    ISIC_2019_data_path = f"{data_path}/ISIC_2019"
    ISIC_csv_path = os.path.join( ISIC_2019_path, "ISIC_2019_core.csv")
    Barcelona_df = pd.read_csv(ISIC_csv_path)
    Barcelona_new = Barcelona_df[[ 'image', 'MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']]
    preprocessed_data = {"columns" : official_columns , "original_columns" : official_columns, "data" : []}

    for i in range(len(Barcelona_new)) :
        temp = list(Barcelona_new.loc[i].values[:-1])
        img_path = os.path.join( ISIC_2019_data_path, temp[0] + ".jpg" )
        origin_labels = temp[1:]
        extended_labels = temp[1:]
        preprocessed_data['data'].append( { 'img_path' : img_path, "origin_labels" : origin_labels, "extended_labels" : extended_labels } )

    file_path = os.path.join( f"{data_path}/ISIC_2019", "ISIC_19_Barcelona.json")

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(preprocessed_data, file, indent="\t")
    ############################################################################################################################


    # Client : Rosendahl
    ############################################################################################################################
    HAM_columns = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC']
    
    HAM_10000_data_path = f"{data_path}/HAM10000"
    rosendahl_df = pd.read_csv(os.path.join( HAM_10000_path, "HAM_ronsendahl.csv"))
    vienna_df = pd.read_csv(os.path.join( HAM_10000_path, "HAM_vienna.csv"))

    ham_labelmap = {
        'akiec' : 'AK', 
        'bcc' : 'BCC', 
        'bkl': 'BKL', 
        'df' : 'DF', 
        'mel' : 'MEL', 
        'nv' : 'NV', 
        'vasc' : 'VASC' }

    preprocessed_data = {"columns" : official_columns, "original_columns" : HAM_columns, "data" : []}

    rosendahl_new = rosendahl_df[['image_id', 'dx']]    
    for i in range(len(rosendahl_new)) :
        img_path = os.path.join( HAM_10000_data_path, rosendahl_new.loc[i]['image_id'] + ".jpg")
        label = ham_labelmap[ rosendahl_new.loc[i]['dx'] ]

        origin_labels = [0] * len(HAM_columns)
        extended_labels = [0] * len(official_columns)

        origin_labels[ HAM_columns.index(label) ] = 1
        extended_labels[ official_columns.index(label) ] = 1

        preprocessed_data['data'].append( { 'img_path' : img_path, "origin_labels" : origin_labels, "extended_labels" :extended_labels } )


    file_path = os.path.join( f"{data_path}/HAM10000", "HAM_rosendahl.json")

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(preprocessed_data, file, indent="\t")
    ############################################################################################################################


    # Client : Vienna
    ############################################################################################################################
    preprocessed_data = {"columns" : official_columns, "original_columns" : HAM_columns, "data" : []}
    vienna_new = vienna_df[['image_id', 'dx']]

    for i in range(len(vienna_new)) :
        img_path = os.path.join( HAM_10000_data_path, vienna_new.loc[i]['image_id'] + ".jpg")
        label = ham_labelmap[ vienna_new.loc[i]['dx'] ]

        origin_labels = [0] * len(HAM_columns)
        extended_labels = [0] * len(official_columns)

        origin_labels[ HAM_columns.index(label) ] = 1
        extended_labels[ official_columns.index(label) ] = 1

        preprocessed_data['data'].append( { 'img_path' : img_path, "origin_labels" : origin_labels, "extended_labels" :extended_labels } )


    file_path = os.path.join( f"{data_path}/HAM10000", "HAM_vienna.json")

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(preprocessed_data, file, indent="\t")
    ############################################################################################################################



    # Client : UFES_brazil
    ############################################################################################################################
    PAD_columns = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'SCC']

    PAD_UFES_20_data_path = f"{data_path}/PAD-UFES-20"
    PAD_UFES_20_df = pd.read_csv( os.path.join( PAD_UFES_20_path, "metadata.csv") )

    # Because seborrheic keratosis(SEK) is in Benign keratosis(BKL), SEK -> BKL
    # Reference : https://challenge.isic-archive.com/landing/2019/ 
    PAD_UFES_20 = {
        'ACK' : 'AK',
        'BCC' : 'BCC', 
        'MEL' : 'MEL', 
        'NEV' : 'NV', 
        'SCC' : 'SCC', 
        'SEK' : 'BKL'
    }
 
    PAD_new = PAD_UFES_20_df[[ 'img_id', 'diagnostic']]
    preprocessed_data = {"columns" : official_columns, "original_columns" : PAD_columns, "data" : []}


    for i in range(len(PAD_new)) :
        img_path = os.path.join( PAD_UFES_20_data_path, PAD_new.loc[i]['img_id'])
        label = PAD_UFES_20[ PAD_new.loc[i]['diagnostic'] ]

        origin_labels = [0] * len(PAD_columns)
        extended_labels = [0] * len(official_columns)

        origin_labels[ PAD_columns.index(label) ] = 1
        extended_labels[ official_columns.index(label) ] = 1

        preprocessed_data['data'].append( { 'img_path' : img_path, "origin_labels" : origin_labels, "extended_labels" :extended_labels } )


    file_path = os.path.join(PAD_UFES_20_path, "PAD_UFES_20.json")

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(preprocessed_data, file, indent="\t")
    ############################################################################################################################



    # Client : SF_canada
    ############################################################################################################################
    Derm7pt_data_path = f"{data_path}/Derm7pt/images"
    Derm7pt_df = pd.read_csv( os.path.join( Derm7pt_path, "meta/meta_core.csv" ) )

    Derm7pt_columns = ['MEL', 'NV', 'BCC', 'BKL', 'DF', 'VASC']

    # reference_1 : https://github.com/jeremykawahara/derm7pt/blob/master/derm7pt/dataset.py
    # reference_2 : https://challenge.isic-archive.com/landing/2019/
    derm7pt_labelmap = {
        'basal cell carcinoma' : 'BCC',
        'blue nevus' : 'NV',
        'clark nevus' : 'NV',
        'combined nevus' : 'NV',
        'congenital nevus' : 'NV',
        'dermal nevus' : 'NV',
        'dermatofibroma' : 'DF', # MISC
        'lentigo' : 'MISC',
        'melanoma' : 'MEL',
        'melanoma (0.76 to 1.5 mm)' : 'MEL',
        'melanoma (in situ)' : 'MEL',
        'melanoma (less than 0.76 mm)' : 'MEL',
        'melanoma (more than 1.5 mm)' : 'MEL',
        'melanoma metastasis' : 'MEL',
        'melanosis' : 'MISC',
        'miscellaneous' : 'MISC',
        'recurrent nevus' : 'NV',
        'reed or spitz nevus' : 'NV',
        'seborrheic keratosis' : 'BKL',
        'vascular lesion' : 'VASC' # 'MISC' 
        }

    Derm7pt_new = Derm7pt_df[['derm','diagnosis']]
    preprocessed_data = {"columns" : official_columns, "original_columns" : Derm7pt_columns, "data" : []}

    for i in range(len(Derm7pt_new)) :
        img_path = os.path.join( Derm7pt_data_path, Derm7pt_new.loc[i]['derm'])
        label = derm7pt_labelmap[ Derm7pt_new.loc[i]['diagnosis'] ]

        origin_labels = [0] * len(Derm7pt_columns)
        extended_labels = [0] * len(official_columns)

        origin_labels[ Derm7pt_columns.index(label) ] = 1
        extended_labels[ official_columns.index(label) ] = 1

        preprocessed_data['data'].append( { 'img_path' : img_path, "origin_labels" : origin_labels, "extended_labels" :extended_labels } )


    file_path = os.path.join(Derm7pt_path, "Derm7pt.json")

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(preprocessed_data, file, indent="\t")

    ############################################################################################################################z