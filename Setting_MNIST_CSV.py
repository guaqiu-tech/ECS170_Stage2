from local_code.base_class.setting import setting

class Setting_MNIST_CSV(setting):
    test_dataset = None

    def set_test_dataset(self, ds):
        self.test_dataset = ds

    def load_run_save_evaluate(self):
        print('loading data...')
        # Load train and test data
        train_data = self.dataset.load()
        test_data = self.test_dataset.load()

        # Pass data to the Method module
        self.method.data = {
            'train': {'X': train_data['X'], 'y': train_data['y']},
            'test': {'X': test_data['X'], 'y': test_data['y']}
        }

        # Run training and testing
        learned_result = self.method.run()

        # Evaluate performance
        print('evaluating performance...')
        self.evaluate.data = learned_result
        score = self.evaluate.evaluate()

        return learned_result, score