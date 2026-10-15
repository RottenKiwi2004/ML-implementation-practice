package math.linear_algebra.error;

import math.linear_algebra.types.BaseLinearAlgebraInterface;

public class ShapeMismatchException extends RuntimeException {
    public ShapeMismatchException(BaseLinearAlgebraInterface a, BaseLinearAlgebraInterface b, String operation) {
        super(String.format("%s %s %s", a.getShapeAsString(), operation, b.getShapeAsString()));
    }
}
