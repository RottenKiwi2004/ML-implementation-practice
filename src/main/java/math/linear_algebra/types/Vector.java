package math.linear_algebra.types;

import math.linear_algebra.error.ShapeMismatchException;

import java.util.Arrays;

public class Vector implements BaseLinearAlgebraInterface {

    protected int n;
    protected double[] entries;

    // Always column vector with dimension N x 1
    private Vector(int n) {
        this.n = n;
        this.entries = new double[n];
    }

    public static Vector zeroes(int n) {
        return new Vector(n);
    }

    public static Vector of(double[] arr) {
        Vector v = new Vector(arr.length);
        v.entries = arr;
        return v;
    }

    public double norm(int p) {
        double sum = 0;
        for (int i=0; i<n; i++) {
            sum += Math.pow(entries[i], p);
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
            sum += this.entries[i] * v2.entries[i];
        }
        return sum;
    }

    public Matrix transpose() {
        Matrix t = Matrix.zeroes(1, this.n);
        for (int i=0; i<n; i++) {
            t.entries[0][i] = this.entries[i];
        }
        return t;
    }

    // Utils
    @Override
    public String toString() {
        return "Vector{" +
                "(" + n +
                " x 1), v=" + Arrays.toString(entries) +
                '}';
    }

    @Override
    public String getShapeAsString() {
        return String.format("(%d x 1)", this.n);
    }
}
