
import matplotlib.pyplot as plt
import numpy as np

def plot_best_error_dynamics_multi_masks(mask_results):
    n_masks = len(mask_results)
    for mask_idx in range(n_masks):
        mask_result = mask_results[mask_idx]
        training_errors = mask_result['training_errors']
        test_errors = mask_result['test_errors']
        best_hyperparams_str = mask_result['best_hyperparams']

        #plt.figure(figsize=(12, 6))

        best_training_error = training_errors[best_hyperparams_str]
        #best_test_error = test_errors[best_hyperparams_str]

        plt.plot(best_training_error, label=f"Training - {best_hyperparams_str}")
        #plt.plot(best_test_error, linestyle='--', label=f"Test - {best_hyperparams_str}")

        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title(f'Mask {mask_idx + 1} - Best Error Dynamics (Best Params: {best_hyperparams_str})')
        plt.legend()
        #plt.show()


def plot_mean_error_dynamics_multi_masks(mask_results):
    n_masks = len(mask_results)
    for mask_idx in range(n_masks):
        mask_result = mask_results[mask_idx]
        training_errors = mask_result['training_errors']
        test_errors = mask_result['test_errors']

        #mean_training_error = np.mean(list(training_errors.values()), axis=0)
        mean_test_error = np.mean(list(test_errors.values()), axis=0)

        #plt.figure(figsize=(12, 6))

        plt.plot(mean_test_error, label=f"Mean Test Error, Mask {mask_idx+1}")
        #plt.plot(mean_test_error, linestyle='--', label="Mean Test Error")

        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title(f'Mask {mask_idx + 1} - Mean Error Dynamics')
        plt.legend()
        #plt.show()

def plot_error_dynamics_multi_masks(mask_results):
    n_masks = len(mask_results)
    for mask_idx in range(n_masks):
        mask_result = mask_results[mask_idx]
        training_errors = mask_result['training_errors']
        test_errors = mask_result['test_errors']
        best_hyperparams_str = mask_result['best_hyperparams']

        plt.figure(figsize=(12, 6))

        for hyperparams_str, training_error in training_errors.items():
            plt.plot(training_error, label=hyperparams_str)

        for hyperparams_str, test_error in test_errors.items():
            plt.plot(test_error, linestyle='--', label=hyperparams_str)

        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title(f'Mask {mask_idx + 1} - Error Dynamics (Best Params: {best_hyperparams_str})')
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        plt.show()


def plot_error_multi_masks(mask_results, masks_names=None, plot_error_type='training', agg_method='mean', by_learning_rule=False, learning_rule=None):
    assert plot_error_type in ['training', 'test'], "plot_error_type must be either 'training' or 'test'."
    assert agg_method in ['mean', 'min', 'median'], "agg_method must be either 'mean', 'min', or 'median'."
    assert not (by_learning_rule and learning_rule), "Cannot specify both by_learning_rule and learning_rule parameters."

    n_masks = len(mask_results)
    if masks_names is None:
        masks_names = ['Mask ' + str(i+1) for i in range(n_masks)]
    learning_rules = ['fa', 'bp']
    
    if by_learning_rule:
        n_plots = len(learning_rules)
    else:
        n_plots = 1

    for plot_idx in range(n_plots):
        if by_learning_rule:
            learning_rule = learning_rules[plot_idx]
            plt.figure(figsize=(12, 6))
        
        for mask_idx in range(n_masks):
            mask_result = mask_results[mask_idx]

            if plot_error_type == 'training':
                errors = mask_result['training_errors']
            else:
                errors = mask_result['test_errors']

            if by_learning_rule:
                filtered_errors = {key: error for key, error in errors.items() if learning_rule in key}
            elif learning_rule:
                filtered_errors = {learning_rule + '_' + plot_error_type: errors[learning_rule + '_' + plot_error_type]}
            else:
                filtered_errors = errors

            if agg_method == 'mean':
                agg_func = np.mean
            elif agg_method == 'min':
                agg_func = np.min
            else:
                agg_func = np.median

            agg_error = agg_func(list(filtered_errors.values()), axis=0)
            plt.plot(agg_error, label=f"Mask {masks_names[mask_idx]}")

        plt.xlabel('Epoch')
        plt.ylabel('Error')
        
        if by_learning_rule:
            plt.title(f'{agg_method.capitalize()} {plot_error_type.capitalize()} Error Dynamics (Learning Rule: {learning_rule.upper()})')
        else:
            plt.title(f'{agg_method.capitalize()} {plot_error_type.capitalize()} Error Dynamics')

        plt.legend()
        plt.show()


    def plot_error_multi_masks2(mask_results, masks_names=None, plot_error_type='training', agg_method='mean', by_learning_rule=False, learning_rule=None, by_n_hidden=False, n_hidden_value=None):
        assert plot_error_type in ['training', 'test'], "plot_error_type must be either 'training' or 'test'."
        assert agg_method in ['mean', 'min', 'median'], "agg_method must be either 'mean', 'min', or 'median'."
        assert not (by_learning_rule and learning_rule), "Cannot specify both by_learning_rule and learning_rule parameters."

        n_masks = len(mask_results)
        if masks_names is None:
            masks_names = ['Mask ' + str(i+1) for i in range(n_masks)]
        learning_rules = ['fa', 'bp']
        
        if by_learning_rule:
            n_plots = len(learning_rules)
        else:
            n_plots = 1

        for plot_idx in range(n_plots):
            if by_learning_rule:
                learning_rule = learning_rules[plot_idx]
                plt.figure(figsize=(12, 6))
            
            for mask_idx in range(n_masks):
                mask_result = mask_results[mask_idx]

                if plot_error_type == 'training':
                    errors = mask_result['training_errors']
                else:
                    errors = mask_result['test_errors']

                if by_learning_rule:
                    filtered_errors = {key: error for key, error in errors.items() if learning_rule in key}
                elif learning_rule:
                    filtered_errors = {learning_rule + '_' + plot_error_type: errors[learning_rule + '_' + plot_error_type]}
                else:
                    filtered_errors = errors

                unique_n_hidden = list(set([int(key.split('_n_hidden')[-1].split('_')[0]) for key in filtered_errors.keys()]))

                for n_hidden in unique_n_hidden:
                    if by_n_hidden or (n_hidden_value is not None and n_hidden == n_hidden_value):
                        n_hidden_errors = {key: error for key, error in filtered_errors.items() if f"n_hidden{n_hidden}" in key}
                    else:
                        continue

                    if agg_method == 'mean':
                        agg_func = np.mean
                    elif agg_method == 'min':
                        agg_func = np.min
                    else:
                        agg_func = np.median

                    agg_error = agg_func(list(n_hidden_errors.values()), axis=0)
                    plt.plot(agg_error, label=f"Mask {masks_names[mask_idx]} (n_hidden={n_hidden})")

            plt.xlabel('Epoch')
            plt.ylabel('Error')
            
            if by_learning_rule:
                plt.title(f'{agg_method.capitalize()} {plot_error_type.capitalize()} Error Dynamics (Learning Rule: {learning_rule.upper()})')
            else:
                plt.title(f'{agg_method.capitalize()} {plot_error_type.capitalize()} Error Dynamics')

            plt.legend()
            plt.show()


  