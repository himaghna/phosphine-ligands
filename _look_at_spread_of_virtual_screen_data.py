from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import yaml



def embed_in_pca_space(X, pca_obj, scaler=None):
    dimensions_returned = [0, 1]
    if scaler:
        X = scaler.transform(X)
    X_in_PCA_space = pca_obj.transform(X)
    return (X_in_PCA_space[:, dimensions_returned[0]],  
            X_in_PCA_space[:, dimensions_returned[1]])

def plot_PC_distribution_of_points(screening_inputs,
                                   screening_color, 
                                   reference_inputs=None,  
                                   reference_color=None):
    concat_inputs = pd.concat([screening_inputs, reference_inputs]).values
    concat_scaler = StandardScaler()
    concat_inputs_normalized = concat_scaler.fit_transform(concat_inputs)
    concat_pca = PCA()
    concat_pca.fit(concat_inputs_normalized)

    screen_pc1, screen_pc2 = embed_in_pca_space(screening_inputs.values, 
                                                scaler=concat_scaler, 
                                                pca_obj=concat_pca)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.scatter(screen_pc1, screen_pc2, 
                c=screening_color, s=60, alpha=0.5, label='Virtual screen')
    if reference_inputs is not None:
        ref_pc1, ref_pc2 = embed_in_pca_space(reference_inputs.values, 
                                              scaler=concat_scaler, 
                                              pca_obj=concat_pca)
        plt.scatter(ref_pc1, ref_pc2, 
                    c=reference_color, s=60, alpha=0.8, label='Reference')
    plt.xlabel('Principal Component 1', fontsize=20)
    plt.ylabel('Principal Component 2', fontsize=20)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config_path', help='Path of configuration yaml file')
    configs = yaml.load(open(parser.parse_args().config_path, "r"),
                        Loader=yaml.FullLoader)
    reference_data_xl = configs.get('reference_data', None)
    reference_color = configs.get('reference_color', None)
    screening_data_xl = configs.get('screening_data')
    screening_color = configs.get('screening_color')
    descriptor_labels_to_use = configs.get('descriptors_to_use')
    
    def get_descriptors_df_from_xl(xl_path, descriptors, sheet_name=0):
        xl_data = pd.read_excel(xl_path, sheet_name=sheet_name, engine='openpyxl')
        descriptor_df = xl_data[descriptors]
        return descriptor_df.fillna(descriptor_df.median())

    screening_inputs = get_descriptors_df_from_xl(
                                screening_data_xl.get('path'), 
                                sheet_name=screening_data_xl.get('sheet_name'),
                                descriptors=descriptor_labels_to_use)
    if reference_data_xl:
        reference_inputs = get_descriptors_df_from_xl(
                                reference_data_xl.get('path'), 
                                sheet_name=reference_data_xl.get('sheet_name'),
                                descriptors=descriptor_labels_to_use)
    else:
        reference_inputs = None
    
    plot_PC_distribution_of_points(screening_inputs=screening_inputs, 
                                   reference_inputs=reference_inputs, 
                                   screening_color=screening_color, 
                                   reference_color=reference_color)
     
