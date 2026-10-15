package com.kiwiredstone;

import math.linear_algebra.types.Vector;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    public static void main() {
        double[] arr = {3.5f, 4.6f, 8.2f, 5.0f};
//        Vector v = Vector.of(arr);
        Vector v = Vector.zeroes(4);
        Vector v2 = Vector.zeroes(4);
//        System.out.println(v2);
        double v3 = v.dot(v2);
        System.out.println(v3);
    }
}
