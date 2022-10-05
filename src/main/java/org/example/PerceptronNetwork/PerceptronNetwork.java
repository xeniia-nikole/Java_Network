package org.example;

/**
 * Hello world!
 *
 */
public class PerceptronNetwork {
    private final int B_BIAS = 0;
    private final int W_WEIGHTS = 0;
    private int layers_num = 0;
    private int[] sizes;

// На вход можно подавать массив любого размера
    public PerceptronNetwork(int... sizes) {
        this.layers_num = sizes.length;
        this.sizes = sizes;
    }

    public static void main(String[] args ) {
        PerceptronNetwork net = new PerceptronNetwork(4, 5, 2);
        int[] x_input = {0, 1, 11, 10};
        int[] z_output = net.feedForward(x_input);
        for (int i = 0; i < z_output.length; i++) {
            System.out.println(z_output[i]);
        }
    }

    private int[] feedForward(int[] x_input) {
        int[] z_output = null;

        for (int i = 1; i < layers_num; i++){
            int size = sizes[i];
            int[] h = new int[size];
            z_output = new int[size];
            for (int j = 1; j < size; j++) {
                for (int k = 1; k < x_input.length; k++) {
                    h[j] += W_WEIGHTS * x_input[k];
                }
                h[j] += B_BIAS;
                z_output[j] = h[j] > 0 ? 1 : 0;
            }
            x_input = z_output;
        }
        return z_output;
    }
}
