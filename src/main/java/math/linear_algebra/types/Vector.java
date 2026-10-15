package math.linear_algebra.types;

import math.linear_algebra.error.ShapeMismatchException;

import java.util.Arrays;

public class Vector implements BaseLinearAlgebraInterface {

    private int n;
    private double[] v;

    // Always column vector with dimension N x 1
    private Vector(int n) {
        this.n = n;
        this.v = new double[n];
    }

    public static Vector zeroes(int n) {
        return new Vector(n);
    }

    public static Vector of(double[] arr) {
        Vector v = new Vector(arr.length);
        v.v = arr;
        return v;
    }

    public double norm(int p) {
        double sum = 0;
        for (int i=0; i<n; i++) {
            sum += Math.pow(v[i], p);
        }
        return sum;
    }

    // Default to Euclidean norm
    public double norm() { return this.norm(2); }

    public double dot(Vector v2) {

        if (this.n != v2.n) {
            throw new ShapeMismatchException(this, v2, "dot");
        }

        double sum = 0;
        for (int i=0; i<n; i++) {
            sum += this.v[i] * v2.v[i];
        }
        return sum;
    }

    // Utils
    @Override
    public String toString() {
        return "Vector{" +
                "n=" + n +
                ", v=" + Arrays.toString(v) +
                '}';
    }

    @Override
    public String getShapeAsString() {
        return String.format("(%d x 1)", this.n);
    }
}
