package com.autograd;

public class ValueTest {

    public static void runTests() {
        testSanityCheck();
        testMoreComplex();
        System.out.println("ValueTest: All tests passed!");
    }

    private static void testSanityCheck() {
        Value x = new Value(-4.0);
        Value z = x.add(2).mul(x).add(3);
        // z = (x + 2) * x + 3
        // forward: (-4 + 2) * -4 + 3 = -2 * -4 + 3 = 8 + 3 = 11
        // backward: dz/dx = d/dx ((x+2)x + 3) = d/dx (x^2 + 2x + 3) = 2x + 2
        // at x=-4: 2(-4) + 2 = -8 + 2 = -6

        z.backward();

        assertDoubleEquals(11.0, z.data, "Forward pass failed");
        assertDoubleEquals(-6.0, x.grad, "Backward pass failed");
        System.out.println("  testSanityCheck passed");
    }

    private static void testMoreComplex() {
        Value a = new Value(-2.0);
        Value b = new Value(3.0);
        Value d = a.mul(b);
        Value e = a.add(b);
        Value f = d.mul(e);

        // f = (a*b) * (a+b)
        // f = a^2*b + a*b^2
        // df/da = 2ab + b^2 = 2(-2)(3) + 3^2 = -12 + 9 = -3
        // df/db = a^2 + 2ab = (-2)^2 + 2(-2)(3) = 4 - 12 = -8

        f.backward();

        assertDoubleEquals(-6.0, d.data, "d data failed");
        assertDoubleEquals(1.0, e.data, "e data failed");
        assertDoubleEquals(-6.0, f.data, "f data failed");

        assertDoubleEquals(-3.0, a.grad, "a grad failed");
        assertDoubleEquals(-8.0, b.grad, "b grad failed");
        System.out.println("  testMoreComplex passed");
    }

    private static void assertDoubleEquals(double expected, double actual, String message) {
        if (Math.abs(expected - actual) > 1e-6) {
            throw new RuntimeException(message + ": expected " + expected + ", got " + actual);
        }
    }
}
