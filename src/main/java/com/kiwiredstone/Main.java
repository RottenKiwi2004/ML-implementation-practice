package com.kiwiredstone;

import math.linear_algebra.error.ShapeMismatchException;
import math.linear_algebra.types.Matrix;
import math.linear_algebra.types.Vector;

import java.lang.reflect.Field;
import java.util.Arrays;

public class Main {

    private static final double EPS = 1e-9;
    private static int passed = 0, failed = 0;

    public static void main() {
        section("Matrix x Matrix");
        test("identity x identity = identity", Main::identityTimesIdentity);
        test("A x I = A (non-square A)", Main::rightIdentity);
        test("I x A = A (non-square A)", Main::leftIdentity);
        test("2x2 known product", Main::known2x2);
        test("2x3 x 3x2 known product", Main::knownNonSquare);
        test("result shape is (m x p)", Main::resultShape);
        test("row x column = 1x1 inner product", Main::rowTimesColumn);
        test("column x row = outer product", Main::columnTimesRow);
        test("zero matrix annihilates", Main::zeroTimesA);
        test("negative and fractional entries", Main::negativeAndFractional);
        test("1x1 x 1x1 is scalar multiplication", Main::scalarCase);
        test("matmul is not commutative", Main::notCommutative);
        test("matmul is associative", Main::associative);
        test("inputs are not mutated", Main::inputsNotMutated);
        test("A x A with A = A^2 (aliasing)", Main::selfMultiply);
        test("shape mismatch throws", Main::shapeMismatchThrows);
        test("shape mismatch message", Main::shapeMismatchMessage);

        section("Matrix x Vector");
        test("I x v = v", Main::identityTimesVector);
        test("2x2 known product", Main::matVecKnown2x2);
        test("2x3 x 3-vector gives 2-vector", Main::matVecWide);
        test("3x2 x 2-vector gives 3-vector", Main::matVecTall);
        test("result shape is (m x 1)", Main::matVecShape);
        test("1x1 x 1-vector", Main::matVecScalar);
        test("zero matrix gives zero vector", Main::matVecZeroMatrix);
        test("A x zero vector = zero vector", Main::matVecZeroVector);
        test("picks out a column with a basis vector", Main::matVecBasisVector);
        test("agrees with A x (n x 1 matrix)", Main::matVecAgreesWithMatMat);
        test("(AB)v = A(Bv)", Main::matVecAssociative);
        test("(A + B)v = Av + Bv", Main::matVecDistributes);
        test("inputs are not mutated", Main::matVecInputsNotMutated);
        test("shape mismatch throws", Main::matVecMismatchThrows);
        test("shape mismatch message", Main::matVecMismatchMessage);

        section("Matrix + Matrix");
        test("2x2 known sum", Main::addKnown2x2);
        test("2x3 known sum (non-square)", Main::addNonSquare);
        test("A + 0 = A", Main::addZero);
        test("A + B = B + A", Main::addCommutative);
        test("(A + B) + C = A + (B + C)", Main::addAssociative);
        test("A + A (aliasing)", Main::addSelf);
        test("result shape is (m x n)", Main::addShape);
        test("inputs are not mutated", Main::addInputsNotMutated);
        test("mismatch throws: both dims differ", Main::addMismatchBoth);
        test("mismatch throws: only rows differ", Main::addMismatchRows);
        test("mismatch throws: only cols differ", Main::addMismatchCols);
        test("shape mismatch message", Main::addMismatchMessage);

        section("Matrix - Matrix");
        test("2x2 known difference", Main::subKnown2x2);
        test("2x3 known difference (non-square)", Main::subNonSquare);
        test("A - 0 = A", Main::subZero);
        test("0 - A = -A", Main::zeroSubA);
        test("A - A = 0", Main::subSelf);
        test("A - B = -(B - A)", Main::subAntiCommutative);
        test("(A + B) - B = A", Main::subUndoesAdd);
        test("inputs are not mutated", Main::subInputsNotMutated);
        test("mismatch throws: both dims differ", Main::subMismatchBoth);
        test("mismatch throws: only rows differ", Main::subMismatchRows);
        test("mismatch throws: only cols differ", Main::subMismatchCols);
        test("shape mismatch message", Main::subMismatchMessage);

        section("Mixed");
        test("A(B + C) = AB + AC", Main::leftDistributive);
        test("(A + B)C = AC + BC", Main::rightDistributive);
        test("A(B - C) = AB - AC", Main::leftDistributiveSub);

        System.out.printf("%n%d passed, %d failed%n", passed, failed);
    }

    // ---------- Matrix x Matrix ----------

    private static void identityTimesIdentity() {
        Matrix i = Matrix.identity(3);
        assertMatrix(new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, i.multiply(Matrix.identity(3)));
    }

    private static void rightIdentity() {
        double[][] a = {{1, 2, 3}, {4, 5, 6}};
        assertMatrix(a, of(a).multiply(Matrix.identity(3)));
    }

    private static void leftIdentity() {
        double[][] a = {{1, 2, 3}, {4, 5, 6}};
        assertMatrix(a, Matrix.identity(2).multiply(of(a)));
    }

    private static void known2x2() {
        Matrix a = of(new double[][]{{1, 2}, {3, 4}});
        Matrix b = of(new double[][]{{5, 6}, {7, 8}});
        assertMatrix(new double[][]{{19, 22}, {43, 50}}, a.multiply(b));
    }

    private static void knownNonSquare() {
        Matrix a = of(new double[][]{{1, 2, 3}, {4, 5, 6}});
        Matrix b = of(new double[][]{{7, 8}, {9, 10}, {11, 12}});
        assertMatrix(new double[][]{{58, 64}, {139, 154}}, a.multiply(b));
    }

    private static void resultShape() {
        Matrix c = Matrix.zeroes(2, 3).multiply(Matrix.zeroes(3, 4));
        assertEquals("(2 x 4)", c.getShapeAsString());
    }

    private static void rowTimesColumn() {
        Matrix row = of(new double[][]{{1, 2, 3}});
        Matrix col = of(new double[][]{{4}, {5}, {6}});
        assertMatrix(new double[][]{{32}}, row.multiply(col));
    }

    private static void columnTimesRow() {
        Matrix col = of(new double[][]{{1}, {2}, {3}});
        Matrix row = of(new double[][]{{4, 5, 6}});
        assertMatrix(new double[][]{{4, 5, 6}, {8, 10, 12}, {12, 15, 18}}, col.multiply(row));
    }

    private static void zeroTimesA() {
        Matrix a = of(new double[][]{{1, 2}, {3, 4}, {5, 6}});
        assertMatrix(new double[2][2], Matrix.zeroes(2, 3).multiply(a));
    }

    private static void negativeAndFractional() {
        Matrix a = of(new double[][]{{-1, 0.5}, {2, -3}});
        Matrix b = of(new double[][]{{4, -2}, {0.25, 1}});
        // [-4 + 0.125, 2 + 0.5], [8 - 0.75, -4 - 3]
        assertMatrix(new double[][]{{-3.875, 2.5}, {7.25, -7}}, a.multiply(b));
    }

    private static void scalarCase() {
        assertMatrix(new double[][]{{-21}}, of(new double[][]{{3}}).multiply(of(new double[][]{{-7}})));
    }

    private static void notCommutative() {
        Matrix a = of(new double[][]{{1, 2}, {3, 4}});
        Matrix b = of(new double[][]{{0, 1}, {1, 0}});
        assertMatrix(new double[][]{{2, 1}, {4, 3}}, a.multiply(b)); // swaps columns
        assertMatrix(new double[][]{{3, 4}, {1, 2}}, b.multiply(a)); // swaps rows
    }

    private static void associative() {
        Matrix a = of(new double[][]{{1, 2}, {3, 4}, {5, 6}});      // 3x2
        Matrix b = of(new double[][]{{1, 0, -1}, {2, 1, 0}});       // 2x3
        Matrix c = of(new double[][]{{2, 1}, {0, 3}, {1, 1}});      // 3x2
        assertMatrix(entries(a.multiply(b).multiply(c)), a.multiply(b.multiply(c)));
    }

    private static void inputsNotMutated() {
        double[][] aVals = {{1, 2}, {3, 4}};
        double[][] bVals = {{5, 6}, {7, 8}};
        Matrix a = of(aVals), b = of(bVals);
        a.multiply(b);
        assertMatrix(aVals, a);
        assertMatrix(bVals, b);
    }

    private static void selfMultiply() {
        Matrix a = of(new double[][]{{1, 1}, {1, 0}}); // Fibonacci matrix
        assertMatrix(new double[][]{{2, 1}, {1, 1}}, a.multiply(a));
    }

    private static void shapeMismatchThrows() {
        assertThrowsMismatch(() -> Matrix.zeroes(2, 3).multiply(Matrix.zeroes(2, 3)));
    }

    private static void shapeMismatchMessage() {
        ShapeMismatchException e = assertThrowsMismatch(() -> Matrix.zeroes(2, 3).multiply(Matrix.zeroes(4, 5)));
        assertEquals("(2 x 3) mat mul (4 x 5)", e.getMessage());
    }

    // ---------- Matrix x Vector ----------

    private static void identityTimesVector() {
        double[] v = {1, -2, 3.5};
        assertVector(v, Matrix.identity(3).multiply(vec(v)));
    }

    private static void matVecKnown2x2() {
        Matrix a = of(new double[][]{{1, 2}, {3, 4}});
        assertVector(new double[]{17, 39}, a.multiply(vec(5, 6)));
    }

    private static void matVecWide() {
        // Output length is the number of ROWS (2), not the number of columns (3)
        Matrix a = of(new double[][]{{1, 2, 3}, {4, 5, 6}});
        assertVector(new double[]{-2, -2}, a.multiply(vec(1, 0, -1)));
    }

    private static void matVecTall() {
        Matrix a = of(new double[][]{{1, 2}, {3, 4}, {5, 6}});
        assertVector(new double[]{-1, -1, -1}, a.multiply(vec(1, -1)));
    }

    private static void matVecShape() {
        Vector v = Matrix.zeroes(4, 2).multiply(Vector.zeroes(2));
        assertEquals("(4 x 1)", v.getShapeAsString());
    }

    private static void matVecScalar() {
        assertVector(new double[]{-12}, of(new double[][]{{4}}).multiply(vec(-3)));
    }

    private static void matVecZeroMatrix() {
        assertVector(new double[3], Matrix.zeroes(3, 2).multiply(vec(7, 8)));
    }

    private static void matVecZeroVector() {
        Matrix a = of(new double[][]{{1, 2, 3}, {4, 5, 6}});
        assertVector(new double[2], a.multiply(Vector.zeroes(3)));
    }

    private static void matVecBasisVector() {
        // A e_j = j-th column of A
        Matrix a = of(new double[][]{{1, 2, 3}, {4, 5, 6}});
        assertVector(new double[]{2, 5}, a.multiply(vec(0, 1, 0)));
    }

    private static void matVecAgreesWithMatMat() {
        Matrix a = of(new double[][]{{2, -1, 0}, {1, 3, 4}});
        double[] v = {3, 1, -2};
        double[][] asColumn = entries(a.multiply(of(new double[][]{{3}, {1}, {-2}})));
        assertVector(new double[]{asColumn[0][0], asColumn[1][0]}, a.multiply(vec(v)));
    }

    private static void matVecAssociative() {
        Matrix a = of(new double[][]{{1, 2}, {0, 1}, {3, -1}});  // 3x2
        Matrix b = of(new double[][]{{2, 0, 1}, {1, 1, 0}});     // 2x3
        Vector v = vec(1, 2, 3);
        assertVector(entries(a.multiply(b.multiply(v))), a.multiply(b).multiply(v));
    }

    private static void matVecDistributes() {
        Matrix a = of(new double[][]{{1, 2}, {3, 4}});
        Matrix b = of(new double[][]{{-1, 0}, {5, 2}});
        Vector v = vec(2, -1);
        double[] av = entries(a.multiply(v)), bv = entries(b.multiply(v));
        assertVector(new double[]{av[0] + bv[0], av[1] + bv[1]}, a.add(b).multiply(v));
    }

    private static void matVecInputsNotMutated() {
        double[][] aVals = {{1, 2}, {3, 4}};
        Matrix a = of(aVals);
        Vector v = vec(5, 6);
        a.multiply(v);
        assertMatrix(aVals, a);
        assertVector(new double[]{5, 6}, v);
    }

    private static void matVecMismatchThrows() {
        assertThrowsMismatch(() -> Matrix.zeroes(2, 3).multiply(Vector.zeroes(2)));
    }

    private static void matVecMismatchMessage() {
        ShapeMismatchException e = assertThrowsMismatch(() -> Matrix.zeroes(2, 3).multiply(Vector.zeroes(4)));
        assertEquals("(2 x 3) mat mul (4 x 1)", e.getMessage());
    }

    // ---------- Matrix + Matrix ----------

    private static void addKnown2x2() {
        Matrix a = of(new double[][]{{1, 2}, {3, 4}});
        Matrix b = of(new double[][]{{5, 6}, {7, 8}});
        assertMatrix(new double[][]{{6, 8}, {10, 12}}, a.add(b));
    }

    private static void addNonSquare() {
        Matrix a = of(new double[][]{{1, -2, 3}, {0.5, 0, -6}});
        Matrix b = of(new double[][]{{-1, 2, 0.25}, {1.5, 7, 6}});
        assertMatrix(new double[][]{{0, 0, 3.25}, {2, 7, 0}}, a.add(b));
    }

    private static void addZero() {
        double[][] a = {{1, 2, 3}, {4, 5, 6}};
        assertMatrix(a, of(a).add(Matrix.zeroes(2, 3)));
    }

    private static void addCommutative() {
        Matrix a = of(new double[][]{{1, 2}, {3, 4}, {5, 6}});
        Matrix b = of(new double[][]{{-3, 0}, {2, 9}, {1, -1}});
        assertMatrix(entries(b.add(a)), a.add(b));
    }

    private static void addAssociative() {
        Matrix a = of(new double[][]{{1, 2}, {3, 4}});
        Matrix b = of(new double[][]{{0.1, 0.2}, {0.3, 0.4}});
        Matrix c = of(new double[][]{{-5, 6}, {7, -8}});
        assertMatrix(entries(a.add(b).add(c)), a.add(b.add(c)));
    }

    private static void addSelf() {
        Matrix a = of(new double[][]{{1, -2}, {3, 0}});
        assertMatrix(new double[][]{{2, -4}, {6, 0}}, a.add(a));
    }

    private static void addShape() {
        assertEquals("(3 x 5)", Matrix.zeroes(3, 5).add(Matrix.zeroes(3, 5)).getShapeAsString());
    }

    private static void addInputsNotMutated() {
        double[][] aVals = {{1, 2}, {3, 4}};
        double[][] bVals = {{5, 6}, {7, 8}};
        Matrix a = of(aVals), b = of(bVals);
        a.add(b);
        assertMatrix(aVals, a);
        assertMatrix(bVals, b);
    }

    private static void addMismatchBoth() {
        assertThrowsMismatch(() -> Matrix.zeroes(2, 3).add(Matrix.zeroes(3, 2)));
    }

    private static void addMismatchRows() {
        assertThrowsMismatch(() -> Matrix.zeroes(2, 3).add(Matrix.zeroes(3, 3)));
    }

    private static void addMismatchCols() {
        assertThrowsMismatch(() -> Matrix.zeroes(2, 3).add(Matrix.zeroes(2, 2)));
    }

    private static void addMismatchMessage() {
        ShapeMismatchException e = assertThrowsMismatch(() -> Matrix.zeroes(2, 3).add(Matrix.zeroes(3, 2)));
        assertEquals("(2 x 3) mat add (3 x 2)", e.getMessage());
    }

    // ---------- Matrix - Matrix ----------

    private static void subKnown2x2() {
        Matrix a = of(new double[][]{{5, 6}, {7, 8}});
        Matrix b = of(new double[][]{{1, 2}, {3, 4}});
        assertMatrix(new double[][]{{4, 4}, {4, 4}}, a.subtract(b));
    }

    private static void subNonSquare() {
        Matrix a = of(new double[][]{{1, -2, 3}, {0.5, 0, -6}});
        Matrix b = of(new double[][]{{-1, 2, 0.25}, {1.5, 7, 6}});
        assertMatrix(new double[][]{{2, -4, 2.75}, {-1, -7, -12}}, a.subtract(b));
    }

    private static void subZero() {
        double[][] a = {{1, 2, 3}, {4, 5, 6}};
        assertMatrix(a, of(a).subtract(Matrix.zeroes(2, 3)));
    }

    private static void zeroSubA() {
        Matrix a = of(new double[][]{{1, -2}, {0, 4}});
        assertMatrix(new double[][]{{-1, 2}, {0, -4}}, Matrix.zeroes(2, 2).subtract(a));
    }

    private static void subSelf() {
        Matrix a = of(new double[][]{{1, 2, 3}, {4, 5, 6}});
        assertMatrix(new double[2][3], a.subtract(a));
    }

    private static void subAntiCommutative() {
        Matrix a = of(new double[][]{{1, 2}, {3, 4}});
        Matrix b = of(new double[][]{{10, 0}, {-1, 2}});
        assertMatrix(new double[][]{{-9, 2}, {4, 2}}, a.subtract(b));
        assertMatrix(new double[][]{{9, -2}, {-4, -2}}, b.subtract(a));
    }

    private static void subUndoesAdd() {
        double[][] aVals = {{1.5, -2}, {3, 4}, {0, 7}};
        Matrix b = of(new double[][]{{9, 8}, {-7, 6}, {5, 0.25}});
        assertMatrix(aVals, of(aVals).add(b).subtract(b));
    }

    private static void subInputsNotMutated() {
        double[][] aVals = {{1, 2}, {3, 4}};
        double[][] bVals = {{5, 6}, {7, 8}};
        Matrix a = of(aVals), b = of(bVals);
        a.subtract(b);
        assertMatrix(aVals, a);
        assertMatrix(bVals, b);
    }

    private static void subMismatchBoth() {
        assertThrowsMismatch(() -> Matrix.zeroes(2, 3).subtract(Matrix.zeroes(3, 2)));
    }

    private static void subMismatchRows() {
        assertThrowsMismatch(() -> Matrix.zeroes(2, 3).subtract(Matrix.zeroes(3, 3)));
    }

    private static void subMismatchCols() {
        assertThrowsMismatch(() -> Matrix.zeroes(2, 3).subtract(Matrix.zeroes(2, 2)));
    }

    private static void subMismatchMessage() {
        ShapeMismatchException e = assertThrowsMismatch(() -> Matrix.zeroes(2, 3).subtract(Matrix.zeroes(3, 2)));
        assertEquals("(2 x 3) mat sub (3 x 2)", e.getMessage());
    }

    // ---------- Mixed ----------

    private static void leftDistributive() {
        Matrix a = of(new double[][]{{1, 2}, {3, 4}, {5, 6}});   // 3x2
        Matrix b = of(new double[][]{{1, 0, 2}, {-1, 3, 1}});    // 2x3
        Matrix c = of(new double[][]{{0, 4, -2}, {2, 1, 1}});    // 2x3
        assertMatrix(entries(a.multiply(b).add(a.multiply(c))), a.multiply(b.add(c)));
    }

    private static void rightDistributive() {
        Matrix a = of(new double[][]{{1, 2}, {3, 4}});
        Matrix b = of(new double[][]{{0, -1}, {2, 5}});
        Matrix c = of(new double[][]{{1, 0, 2}, {-1, 3, 1}});    // 2x3
        assertMatrix(entries(a.multiply(c).add(b.multiply(c))), a.add(b).multiply(c));
    }

    private static void leftDistributiveSub() {
        Matrix a = of(new double[][]{{2, -1}, {0, 3}});
        Matrix b = of(new double[][]{{1, 2}, {3, 4}});
        Matrix c = of(new double[][]{{4, 3}, {2, 1}});
        assertMatrix(entries(a.multiply(b).subtract(a.multiply(c))), a.multiply(b.subtract(c)));
    }

    // ---------- Harness ----------

    private static void section(String name) {
        System.out.println("\n== " + name + " ==");
    }

    private static void test(String name, Runnable body) {
        try {
            body.run();
            passed++;
            System.out.println("PASS  " + name);
        } catch (Throwable t) {
            failed++;
            System.out.println("FAIL  " + name + "\n      " + t);
        }
    }

    private static ShapeMismatchException assertThrowsMismatch(Runnable body) {
        try {
            body.run();
        } catch (ShapeMismatchException e) {
            return e;
        }
        throw new AssertionError("expected ShapeMismatchException but nothing was thrown");
    }

    private static void assertEquals(String expected, String actual) {
        if (!expected.equals(actual)) {
            throw new AssertionError("expected " + expected + " but got " + actual);
        }
    }

    private static void assertMatrix(double[][] expected, Matrix actual) {
        String expectedShape = String.format("(%d x %d)", expected.length, expected[0].length);
        assertEquals(expectedShape, actual.getShapeAsString());

        double[][] got = entries(actual);
        boolean ok = got.length == expected.length;
        for (int i = 0; ok && i < expected.length; i++) {
            ok = got[i].length == expected[i].length;
            for (int j = 0; ok && j < expected[i].length; j++) {
                ok = Math.abs(got[i][j] - expected[i][j]) < EPS;
            }
        }
        if (!ok) {
            throw new AssertionError("expected " + Arrays.deepToString(expected)
                    + " but got " + Arrays.deepToString(got));
        }
    }

    private static void assertVector(double[] expected, Vector actual) {
        if (actual == null) {
            throw new AssertionError("expected " + Arrays.toString(expected) + " but got null");
        }
        assertEquals(String.format("(%d x 1)", expected.length), actual.getShapeAsString());

        double[] got = entries(actual);
        boolean ok = got.length == expected.length;
        for (int i = 0; ok && i < expected.length; i++) {
            ok = Math.abs(got[i] - expected[i]) < EPS;
        }
        if (!ok) {
            throw new AssertionError("expected " + Arrays.toString(expected)
                    + " but got " + Arrays.toString(got));
        }
    }

    // Matrix has no public setters/getters, so tests reach into the private `entries` field.
    // Values are written into the array created by Matrix.zeroes, so the constructor is still exercised.
    private static Matrix of(double[][] values) {
        Matrix a = Matrix.zeroes(values.length, values[0].length);
        double[][] e = entries(a);
        for (int i = 0; i < values.length; i++) {
            System.arraycopy(values[i], 0, e[i], 0, values[i].length);
        }
        return a;
    }

    // Copies the input so a test's expected array can't be changed through the Vector.
    private static Vector vec(double... values) {
        return Vector.of(values.clone());
    }

    private static double[][] entries(Matrix a) {
        return (double[][]) readField(Matrix.class, "entries", a);
    }

    private static double[] entries(Vector v) {
        return (double[]) readField(Vector.class, "entries", v);
    }

    private static Object readField(Class<?> cls, String name, Object target) {
        try {
            Field f = cls.getDeclaredField(name);
            f.setAccessible(true);
            return f.get(target);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }
}
