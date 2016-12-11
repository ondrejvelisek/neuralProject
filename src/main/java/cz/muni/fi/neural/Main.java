package cz.muni.fi.neural;

import java.io.*;
import java.util.*;
import java.util.logging.*;

public class Main {
    public static Logger logger = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);



    public static void main(String[] args){
        ConfigReader  mlpConfig = ConfigReader.getInstance();
        DataReader dataReader = new DataReader();
        List<List<Double>> dataMatrix = null;

        try {
            LogManager.getLogManager().readConfiguration(new FileInputStream("src/main/resources/logger.properties"));
            FileHandler logFile = new FileHandler("./debug.log");
            Logger.getLogger(Logger.GLOBAL_LOGGER_NAME).addHandler(logFile);
        } catch (IOException e) {
            e.printStackTrace();
        }

        try {
            dataReader.setFile(mlpConfig.getDataSourceName());
            dataMatrix = dataReader.csvToMatrix();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        dataMatrix = dataReader.normalize(dataMatrix);
        dataReader.splitDataSet(0.6,0.2,0.2,dataMatrix.size());

        List<List<Double>> trainingSet = dataReader.getTrainingSet(dataMatrix);
        List<List<Double>> validationSet = dataReader.getValidationSet(dataMatrix);
        List<List<Double>> testSet = dataReader.getTestSet(dataMatrix);

        Double[][] trainingInputsMatrix = dataReader.getInputsMatrixWithBiasInput(trainingSet);
        Double[] trainingOutputsVector = dataReader.getOutputVector(trainingSet);
        System.out.println(trainingInputsMatrix.length);
        Double[][] validationInputsMatrix = dataReader.getInputsMatrixWithBiasInput(validationSet);
        Double[] validationOutputsVector = dataReader.getOutputVector(validationSet);
        System.out.println(validationInputsMatrix.length);

        Double[][] testInputsMatrix = dataReader.getInputsMatrixWithBiasInput(testSet);
        Double[] testOutputsVector = dataReader.getOutputVector(testSet);
        System.out.println(testInputsMatrix.length);

//        if(mlpConfig.loadedDatasetDebug()) {
//            for (int i = 0; i < inputsMatrix.length; i++) {
//                System.out.print(i + " ");
//                for (int j = 0; j < inputsMatrix[1].length; j++) {
//                    System.out.print(inputsMatrix[i][j] + " ");
//                }
//                System.out.println();
//            }
//            for (int i = 0; i < outputsVector.length; i++) {
//                System.out.print(i + " ");
//                System.out.print(outputsVector[i] + " ");
//
//                System.out.println();
//            }
//        }

        List<Integer> layersStructure = new ArrayList<>();
        int inputLayerSize = mlpConfig.getInputVectors().size();
        int outputLayerSize = 1;

        layersStructure.add(inputLayerSize);
        layersStructure.addAll(mlpConfig.getMlpArchitecture());
        layersStructure.add(outputLayerSize);

        NeuralNetwork net = new MultilayerPerceptron(layersStructure);

       // net.learn(inputsMatrix, outputsVector);

    }



}
