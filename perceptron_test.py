from perceptron_multiclass import PerceptronMultiClass
import numpy as np

input_datas = np.array(
                [
                    [0, 0, 1],
                    [0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]
                ] 
                )

input_targets = np.array(
                [
                    [0, 0, 0, 1], 
                    [0, 0, 1, 0], 
                    [0, 1, 0, 0], 
                    [1, 0, 0, 0]
                ])


def test():
    p = PerceptronMultiClass(2, 4)
    p.train(input_datas, input_targets, 10000)
    
    print("Tesing with trained data")
    p.test(input_datas, input_targets)

    print("Tesing with un-trained data")
    test_datas = np.array(
                [
                    [0.3, 0.3, 1],
                    [0.3, 1,   1],
                    [0.8, 0,   1],
                    [0.8, 0.8, 1]
                ] 
                )

    test_targets = np.array(
                [
                    [0, 0, 0, 1], 
                    [0, 0, 1, 0], 
                    [0, 1, 0, 0], 
                    [1, 0, 0, 0]
                ])
     
    p.test(test_datas, test_targets) 

    # p.write_weights("weight_test.txt")
    print("Traing with more_data")
    input_datas_2 = np.array(
                [
                    [0.1, 0.2, 1],
                    [0.2, 0.9, 1],
                    [0.8, 0.2, 1],
                    [0.9, 0.8, 1]
                ] 
                )

    input_targets_2 = np.array(
                [
                    [0, 0, 0, 1], 
                    [0, 0, 1, 0], 
                    [0, 1, 0, 0], 
                    [1, 0, 0, 0]
                ])

    p.train(np.append(input_datas, input_datas_2, axis=0), 
            np.append(input_targets, input_targets_2, axis=0), 100000)
    p.test(np.append(input_datas, input_datas_2, axis=0), 
            np.append(input_targets, input_targets_2, axis=0)) 

    p.test(test_datas, test_targets) 


if __name__ == '__main__':
    test()
