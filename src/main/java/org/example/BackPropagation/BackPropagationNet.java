package org.example.BackPropagation;

import org.jblas.DoubleMatrix;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

public class BackPropagationNet{
    double W = 2;
    private int layersNum;
    private int[] sizes;
    private DoubleMatrix[] weights;
    private DoubleMatrix[] biases;
    // На вход можно подавать массив любого размера
    public BackPropagationNet(int @NotNull ... sizes) {
        this.sizes = sizes;
        this.layersNum = sizes.length;

        this.biases = new DoubleMatrix[sizes.length - 1];
        this.weights = new DoubleMatrix[sizes.length - 1];

        // biases
        for (int i = 1; i < sizes.length; i++) {
            double[][] temp = new double[sizes[i]][];
            for (int j = 0; j < sizes[i]; j++) {
                double[] ins_temp = new double[]{ -1 };
                temp[j] = ins_temp;
            }
            biases[i - 1] = new DoubleMatrix(temp);
        }
// weights
        for (int i = 1; i < sizes.length; i++) {
            double[][] temp = new double[sizes[i]][];
            for (int j = 0; j < sizes[i]; j++) {
                double[] ins2_temp = new double[sizes[i-1]];
                for (int k = 0; k < sizes[i-1]; k++) {
                    ins2_temp[k] = 1;
                }
                temp[j] = ins2_temp;
            }
            weights[i-1] = new DoubleMatrix(temp);
        }
//        double[][] customWeights = new double[][] {
//                {0, W, 0, W, 0, W, 0, W, 0, W},
//                {0, 0, W, W, 0, 0, W, W, 0, 0},
//                {0, 0, 0, 0, W, W, W, W, 0, 0},
//                {0, 0, 0, 0, 0, 0, 0, 0, W, W}
//        };
//        weights[0] = new DoubleMatrix(customWeights);
//        double[][] customWeights = new double[][]{
//                {0, W, 0, W, 0, W, W, W, W, W},
//                {0, 0, W, W, 0, 0, 0, W, 0, 0},
//                {0, 0, 0, 0, W, W, W, W, 0, 0},
//                {0, 0, 0, 0, 0, 0, 0, 0, 0, W}
//        };
//        w_weights[0] = new DoubleMatrix(customWeights);

    }

    public static void main(String[] args ) {
        BackPropagationNet net = new BackPropagationNet(1, 1);
        double[] inputs = { 0 };
        double[] outputs = { 0 };
        double[][] inputsOuputs = new double[][] { inputs, outputs };
        DoubleMatrix[][] deltas = net.backProp(inputsOuputs);
        for (int i = 0; i < net.biases.length; i++) {
            net.biases[i] = net.biases[i].sub(deltas[0][i].mul(4));
        }
        System.out.println("Complete");
    }

    private DoubleMatrix[][] backProp(double[][] inputsOuputs) {
        DoubleMatrix[] nablaB = new DoubleMatrix[biases.length];
        DoubleMatrix[] nablaW = new DoubleMatrix[weights.length];

        for (int i = 0; i < nablaB.length; i++) {
            nablaB[i] = new DoubleMatrix(biases[i].getRows(), biases[i].getColumns());
        }
        for (int i = 0; i < nablaW.length; i++) {
            nablaW[i] = new DoubleMatrix(weights[i].getRows(), weights[i].getColumns());
        }

        // FeedForward
        DoubleMatrix activation = new DoubleMatrix(inputsOuputs[0]);
        DoubleMatrix[] activations = new DoubleMatrix[layersNum];
        activations[0] = activation;
        DoubleMatrix[] zs = new DoubleMatrix[layersNum - 1];

        for (int i = 0; i < layersNum - 1; i++) {
            double[] scalars = new double[weights[i].rows];
            for (int j = 0; j < weights[i].rows; j++) {
                scalars[j] = weights[i].getRow(j).dot(activation) + biases[i].get(j);
            }
            DoubleMatrix z = new DoubleMatrix(scalars);
            zs[i] = z;
            activation = sigmoid(z);
            activations[i + 1] = activation;
        }

        // Backward pass
        DoubleMatrix output = new DoubleMatrix(inputsOuputs[1]);
        DoubleMatrix delta = costDerivative(activations[activations.length - 1], output)
                .mul(sigmoidPrime(zs[zs.length - 1])); // BP1
        nablaB[nablaB.length - 1] = delta; // BP3
        nablaW[nablaW.length - 1] = delta.mmul(activations[activations.length - 2].transpose()); // BP4
        for (int layer = 2; layer < layersNum; layer++) {
            DoubleMatrix z = zs[zs.length - layer];
            DoubleMatrix sp = sigmoidPrime(z);
            delta = weights[weights.length + 1 - layer].transpose().mmul(delta).mul(sp); // BP2
            nablaB[nablaB.length - layer] = delta; // BP3
            nablaW[nablaW.length - layer] = delta.mmul(activations[activations.length - 1 - layer].transpose()); // BP4
        }
        return new DoubleMatrix[][] { nablaB, nablaW };
    }

    private DoubleMatrix sigmoidPrime(DoubleMatrix z) {
        return sigmoid(z).mul(sigmoid(z).rsub(1));
    }

    private DoubleMatrix costDerivative(DoubleMatrix outputActivations, DoubleMatrix output) {
        return outputActivations.sub(output);
    }

    private DoubleMatrix feedForward(DoubleMatrix doubleMatrix) {
        for (int i = 0; i < layersNum - 1; i++){
            double[] temp = new double[weights[i].rows];
            for (int j = 0; j < weights[i].rows; j++) {
                temp[j] = weights[i].getRow(j).dot(doubleMatrix) + biases[i].get(j);
            }
            DoubleMatrix z_output = new DoubleMatrix(temp);
            doubleMatrix = sigmoid(z_output);
        }
        return doubleMatrix;
    }

    @Contract("_ -> new")
    private @NotNull DoubleMatrix sigmoid(@NotNull DoubleMatrix z) {
        double[] output = new double[z.length];
        for (int i = 0; i < output.length; i++) {
            output[i] = 1 / (1 + Math.exp(-z.get(i)));
        }
        return new DoubleMatrix(output);
    }

}
