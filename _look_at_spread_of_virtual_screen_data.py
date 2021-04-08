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


def show_double_bar_plot(heights_class1, 
                         heights_class2, 
                         legend_class1, legend_class2,
                         x_positions, x_labels, y_label,
                         color_class1, 
                         color_class2):

    bar_width = 0.27  
    fig = plt.figure()
    ax = fig.add_subplot(111)

    bars_class1 = ax.bar(x_positions, 
                         heights_class1, 
                         bar_width, 
                         color=color_class1)
    bars_class2 = ax.bar(x_positions, 
                         heights_class2, 
                         bar_width, 
                         color=color_class1)

    ax.set_ylabel(y_label, fontsize=20)
    ax.set_xticks(x_positions + width)
    ax.set_xticklabels(x_labels, fontsize=20)
    ax.legend((bars_class1[0], bars_class2[0]), (legend_class1, legend_class2), 
              fontsize=24)

    def autolabel(all_bars):
        for bar in all_bars:
            h = bar.get_height()
            ax.text(rect.get_x()+bar.get_width()/2., 1.05*h, '%d'%int(h),
                    ha='center', va='bottom')

    autolabel(bars_class1)
    autolabel(bars_class2)
    plt.show()

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

def plot_difference_in_means(screening_inputs, 
                             reference_inputs, 
                             screening_color,
                             reference_color):
    for column_name in screening_inputs.columns:
    
    show_double_bar_plot(heights_class1, 
                         heights_class2, 
                         legend_class1, legend_class2,
                         x_positions, x_labels, y_label,
                         color_class1, 
                         color_class2





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
    import pickle
    pickle.dump(screening_inputs.values, open('X_virtual_screen.p', "wb"))
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

    plot_difference_in_means(screening_inputs=screening_inputs, 
                             reference_inputs=reference_inputs, 
                             screening_color=screening_color, 
                             reference_color=reference_color)
     
