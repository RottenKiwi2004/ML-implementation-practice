package math.linear_algebra.types;

import math.linear_algebra.error.ShapeMismatchException;

import java.util.Arrays;

public class Matrix implements BaseLinearAlgebraInterface {

    protected int m, n;
    protected double[][] entries;

    // Always column vector with dimension N x 1
    private Matrix(int m, int n) {
        this.m = m;
        this.n = n;
        this.entries = new double[m][n];
    }

    public static Matrix identity(int n) {
        Matrix a = new Matrix(n, n);
        for(int i=0; i<n; i++) {
            a.entries[i][i] = 1.0;
        }
        return a;
    }

    public static Matrix zeroes(int m, int n) {
        return new Matrix(m, n);
    }

    public Matrix multiply(Matrix b) {
        if (this.n != b.m) {
            throw new ShapeMismatchException(this, b, "mat mul");
        }

        int m = this.m, n = b.n;
        Matrix c = Matrix.zeroes(m, n);
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                for (int k=0; k<this.n; k++) {
                    c.entries[i][j] += this.entries[i][k] * b.entries[k][j];
                }
            }
        }

        return c;
    }

    public Vector multiply(Vector b) {
        if (this.n != b.n) {
            throw new ShapeMismatchException(this, b, "mat mul");
        }

        Vector v = Vector.zeroes(this.m);
        for (int i=0; i<this.m; i++) {
            for (int k=0; k<this.n; k++) {
                v.entries[i] += this.entries[i][k] * b.entries[k];
            }
        }
        return v;
    }

    public Matrix add(Matrix b) {
        if (this.m != b.m || this.n != b.n) {
            throw new ShapeMismatchException(this, b, "mat add");
        }

        Matrix c = Matrix.zeroes(m, n);
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                c.entries[i][j] = this.entries[i][j] + b.entries[i][j];
            }
        }

        return c;
    }

    public Matrix subtract(Matrix b) {
        if (this.m != b.m || this.n != b.n) {
            throw new ShapeMismatchException(this, b, "mat sub");
        }

        Matrix c = Matrix.zeroes(m, n);
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                c.entries[i][j] = this.entries[i][j] - b.entries[i][j];
            }
        }

        return c;
    }

    public Matrix transpose() {
        Matrix t = Matrix.zeroes(this.n, this.m);
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                t.entries[j][i] = this.entries[i][j];
            }
        }
        return t;
    }

    @Override
    public String toString() {
        return "Matrix{" +
                "(" + m +
                " x " + n +
                "), entries=" + Arrays.toString(entries) +
                '}';
    }

    @Override
    public String getShapeAsString() {
        return String.format("(%d x %d)", m, n);
    }
}
