package org.example.NumberToBinaryV2;

import org.jblas.DoubleMatrix;
import org.jblas.util.Random;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

public class SigmoidNetwork {
    private int layersNum;
    private int[] sizes;
    private DoubleMatrix[] weights;
    private DoubleMatrix[] biases;

    // Any size array can be input
    public SigmoidNetwork(int @NotNull ... sizes) {
        this.sizes = sizes;
        this.layersNum = sizes.length;

        this.biases = new DoubleMatrix[sizes.length - 1];
        this.weights = new DoubleMatrix[sizes.length - 1];

        // biases
        for (int i = 1; i < sizes.length; i++) {
            double[][] temp = new double[sizes[i]][];
            for (int j = 0; j < sizes[i]; j++) {
                double[] b = new double[] { Random.nextGaussian() };
                temp[j] = b;
            }
            biases[i - 1] = new DoubleMatrix(temp);
        }
        // weights
        for (int i = 1; i < sizes.length; i++) {
            double[][] temp = new double[sizes[i]][];
            for (int j = 0; j < sizes[i]; j++) {
                double[] w = new double[sizes[i - 1]];
                for (int k = 0; k < sizes[i - 1]; k++) {
                    w[k] = Random.nextGaussian();
                }
                temp[j] = w;
            }
            weights[i - 1] = new DoubleMatrix(temp);
        }

    }

    public static void main(String[] args) {
        List<double[][]> inputsOutputs = new ArrayList<>();
        for (int i = 0; i < 256; i++) {
            double[][] io = new double[2][];
            double[] x;
            double[] y;

            String binary = String.format("%8s", Integer.toBinaryString(i)).replace(' ', '0');
            x = Stream.iterate(0, n -> 0).limit(256).mapToDouble(num -> (double)num).toArray();
            y = Arrays.stream(binary.split("")).mapToDouble(Double::parseDouble).toArray();

            x[i] = 1;
            io[0] = x;
            io[1] = y;
            inputsOutputs.add(io);
        }
        SigmoidNetwork net = new SigmoidNetwork(256, 64, 8);
        net.SGD(inputsOutputs, 100, 8, 15, inputsOutputs.subList(0, 100));
    }

    /**
     *
     * @param trainingData  - list of arrays (x, y) representing the training inputs
     *                      and corresponding desired outputs
     * @param epochs        - the number of epochs to train for
     * @param miniBatchSize - the size of the mini-batches to use when sampling
     * @param eta           - the learning rate, η
     * @param testData      - the test data use to evaluate the net
     * @return void
     */
    public void SGD(@NotNull List<double[][]> trainingData, int epochs, int miniBatchSize, double eta,
                    List<double[][]> testData) {

        int nTest = 0;
        int n = trainingData.size();

        if (testData != null) {
            nTest = testData.size();
        }

        for (int j = 0; j < epochs; j++) {
            Collections.shuffle(trainingData);
            List<List<double[][]>> miniBatches = new ArrayList<>();
            for (int k = 0; k < n; k += miniBatchSize) {
                miniBatches.add(trainingData.subList(k, k + miniBatchSize));
            }
            for (List<double[][]> miniBatch : miniBatches) {
                updateMiniBatch(miniBatch, eta);
            }

            if (testData != null) {
                int e = evaluate(testData);
                System.out.println(String.format("Epoch %d: %d / %d", j, e, nTest));
            } else {
                System.out.println(String.format("Epoch %d complete", j));
            }
        }

    }

    /**
     *
     * @param testData - the test data used to evaluate the net
     * @return the number of test inputs for which the neural network outputs the
     *         correct result
     */
    private int evaluate(@NotNull List<double[][]> testData) {
        int sum = 0;
        for (double[][] inputOutput : testData) {
            DoubleMatrix x = new DoubleMatrix(inputOutput[0]);
            DoubleMatrix y = new DoubleMatrix(inputOutput[1]);
            DoubleMatrix netOutput = feedForward(x);
            StringBuilder sb = new StringBuilder();
            StringBuilder sb2 = new StringBuilder();
            for (double d : netOutput.toArray()) {
                sb.append(d >= 0.5 ? 1 : 0);
            }
            for (double d : y.toArray()) {
                sb2.append(d >= 0.5 ? 1 : 0);
            }
            if (Integer.parseInt(sb.toString(), 2) == Integer.parseInt(sb2.toString(), 2)) {
                sum++;
            }
        }
        return sum;
    }

    /**
     *
     * Update the network’s weights and biases by applying gradient descent using
     * backpropagation to a single mini batch. The "mini_batch" is a list of arrays
     * "(x, y)", and "eta" is the learning rate.
     *
     * @param miniBatch - part of a training data
     * @param eta       - the learning rate
     */
    private void updateMiniBatch(List<double[][]> miniBatch, double eta) {
        DoubleMatrix[] nablaB = new DoubleMatrix[biases.length];
        DoubleMatrix[] nablaW = new DoubleMatrix[weights.length];

        for (int i = 0; i < nablaB.length; i++) {
            nablaB[i] = new DoubleMatrix(biases[i].getRows(), biases[i].getColumns());
        }
        for (int i = 0; i < nablaW.length; i++) {
            nablaW[i] = new DoubleMatrix(weights[i].getRows(), weights[i].getColumns());
        }

        for (double[][] inputOutput : miniBatch) {
            DoubleMatrix[][] deltas = backProp(inputOutput);

            DoubleMatrix[] deltaNablaB = deltas[0];
            DoubleMatrix[] deltaNablaW = deltas[1];

            for (int i = 0; i < nablaB.length; i++) {
                nablaB[i] = nablaB[i].add(deltaNablaB[i]);
            }
            for (int i = 0; i < nablaW.length; i++) {
                nablaW[i] = nablaW[i].add(deltaNablaW[i]);
            }
        }
        for (int i = 0; i < biases.length; i++) {
            biases[i] = biases[i].sub(nablaB[i].mul(eta / miniBatch.size()));
        }
        for (int i = 0; i < weights.length; i++) {
            weights[i] = weights[i].sub(nablaW[i].mul(eta / miniBatch.size()));
        }
    }

    /**
     *
     * Return an array (nablaB , nablaW) representing the gradient for the cost
     * function C. "nablaB" and "nablaW" are layer-by-layer arrays of DoubleMatrices
     * , similar to this.biases and this.weights.
     *
     * @param inputsOutputs
     * @return
     */
    @Contract("_ -> new")
    private @NotNull DoubleMatrix[][] backProp( @NotNull double[][] inputsOutputs) {
        DoubleMatrix[] nablaB = new DoubleMatrix[biases.length];
        DoubleMatrix[] nablaW = new DoubleMatrix[weights.length];

        for (int i = 0; i < nablaB.length; i++) {
            nablaB[i] = new DoubleMatrix(biases[i].getRows(), biases[i].getColumns());
        }
        for (int i = 0; i < nablaW.length; i++) {
            nablaW[i] = new DoubleMatrix(weights[i].getRows(), weights[i].getColumns());
        }

        // FeedForward
        DoubleMatrix activation = new DoubleMatrix(inputsOutputs[0]);
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
        DoubleMatrix output = new DoubleMatrix(inputsOutputs[1]);
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

    private DoubleMatrix costDerivative(@NotNull DoubleMatrix outputActivations, DoubleMatrix output) {
        return outputActivations.sub(output);
    }

    /**
     * @param doubleMatrix - activation vector - the 1st layer also called the input layer
     * @return DoubleMatrix - vector containing output from the network consisting
     * of float numbers between 0 and 1
     */
    private DoubleMatrix feedForward(DoubleMatrix doubleMatrix) {
        for (int i = 0; i < layersNum - 1; i++) {
            double[] temp = new double[weights[i].rows];
            for (int j = 0; j < weights[i].rows; j++) {
                temp[j] = weights[i].getRow(j).dot(doubleMatrix) + biases[i].get(j);
            }
            DoubleMatrix z_output = new DoubleMatrix(temp);
            doubleMatrix = sigmoid(z_output);
        }
        return doubleMatrix;
    }

    /**
     * @param z - input vector created by finding dot product of weights and inputs
     *          and added a bias of a neuron
     * @return output vector - inputs for the next layer
     */
    @Contract("_ -> new")
    private @NotNull DoubleMatrix sigmoid(@NotNull DoubleMatrix z) {
        double[] output = new double[z.length];
        for (int i = 0; i < output.length; i++) {
            output[i] = 1 / (1 + Math.exp(-z.get(i)));
        }
        return new DoubleMatrix(output);
    }

}

