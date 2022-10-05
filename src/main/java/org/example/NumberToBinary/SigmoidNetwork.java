package org.example.NumberToBinary;

import org.jblas.DoubleMatrix;

public class SigmoidNetwork {
    double W = 2;
    private int layersNum;
    private int[] sizes;
    private DoubleMatrix[] weights;
    private DoubleMatrix[] biases;
    // На вход можно подавать массив любого размера
    public SigmoidNetwork(int... sizes) {
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
//        for (int i = 1; i < sizes.length; i++) {
//            double[][] temp = new double[sizes[i]][];
//            for (int j = 0; j < sizes[i]; j++) {
//                double[] ins2_temp = new double[sizes[i-1]];
//                for (int k = 0; k < sizes[i-1]; k++) {
//                    ins2_temp[k] = 0;
//                }
//                temp[j] = ins2_temp;
//            }
//            w_weights[i-1] = new DoubleMatrix(temp);
//        }
        double[][] customWeights = new double[][] {
                {0, W, 0, W, 0, W, 0, W, 0, W},
                {0, 0, W, W, 0, 0, W, W, 0, 0},
                {0, 0, 0, 0, W, W, W, W, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, W, W}
        };
        weights[0] = new DoubleMatrix(customWeights);
//        double[][] customWeights = new double[][]{
//                {0, W, 0, W, 0, W, W, W, W, W},
//                {0, 0, W, W, 0, 0, 0, W, 0, 0},
//                {0, 0, 0, 0, W, W, W, W, 0, 0},
//                {0, 0, 0, 0, 0, 0, 0, 0, 0, W}
//        };
//        w_weights[0] = new DoubleMatrix(customWeights);

    }

    public static void main(String[] args ) {
        SigmoidNetwork net = new SigmoidNetwork(10, 4);
        net.testing(net);
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

    private DoubleMatrix sigmoid(DoubleMatrix z) {
        double[] output = new double[z.length];
        for (int i = 0; i < output.length; i++) {
            output[i] = 1 / (1 + Math.exp(-z.get(i)));
        }
        return new DoubleMatrix(output);
    }

    private double[] format(int x) {
        return switch (x) {
            case 0 -> new double[]{1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            case 1 -> new double[]{0, 1, 0, 0, 0, 0, 0, 0, 0, 0};
            case 2 -> new double[]{0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
            case 3 -> new double[]{0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
            case 4 -> new double[]{0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
            case 5 -> new double[]{0, 0, 0, 0, 0, 1, 0, 0, 0, 0};
            case 6 -> new double[]{0, 0, 0, 0, 0, 0, 1, 0, 0, 0};
            case 7 -> new double[]{0, 0, 0, 0, 0, 0, 0, 1, 0, 0};
            case 8 -> new double[]{0, 0, 0, 0, 0, 0, 0, 0, 1, 0};
            case 9 -> new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
            default -> new double[10];
        };
    }

    private void testing(SigmoidNetwork net){

                double[] x_input1 = format(4);
                double[] x_input2 = format(3);
                double[] x_input3 = format(7);
                double[] x_input4 = format(9);
                double[] x_input5 = format(5);

                DoubleMatrix z_output1 = net.feedForward(new DoubleMatrix(x_input1));
                DoubleMatrix z_output2 = net.feedForward(new DoubleMatrix(x_input2));
                DoubleMatrix z_output3 = net.feedForward(new DoubleMatrix(x_input3));
                DoubleMatrix z_output4 = net.feedForward(new DoubleMatrix(x_input4));
                DoubleMatrix z_output5 = net.feedForward(new DoubleMatrix(x_input5));

                // 4 -> 0010 вместо 0100
                System.out.println("0 1 0 0 ->"+z_output1.toString("%.0f"));
                System.out.println("1 1 0 0 ->"+z_output2.toString("%.0f"));
                System.out.println("1 1 1 0 ->"+z_output3.toString("%.0f"));
                System.out.println("1 0 0 1 ->"+z_output4.toString("%.0f"));
                System.out.println("1 0 1 0 ->"+z_output5.toString("%.0f"));

    }

}
