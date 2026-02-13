package com.autograd;

public class TestRunner {
    public static void main(String[] args) {
        System.out.println("Running AutoGrad-Java Tests...");
        try {
            ValueTest.runTests();
            MicroGPTTest.runTests();
            System.out.println("\nSUCCESS: All tests passed.");
        } catch (Exception e) {
            System.err.println("\nFAILURE: Tests failed.");
            e.printStackTrace();
            System.exit(1);
        }
    }
}
