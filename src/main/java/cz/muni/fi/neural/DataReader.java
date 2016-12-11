package cz.muni.fi.neural;

import au.com.bytecode.opencsv.CSVReader;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by Simon on 06.11.2016.
 */
public class DataReader {
    private CSVReader reader;
    private List<Integer> trainingSetIndices;
    private List<Integer> validationSetIndices;
    private List<Integer> testSetIndices;

    public DataReader(){
    }

    public void setFile(String fileName) throws FileNotFoundException {
        ConfigReader mlpConfig = ConfigReader.getInstance();
        reader = new CSVReader(new FileReader(fileName),mlpConfig.getCsvSeparator());
    }

    public List<List<Double>> csvToMatrix() throws IOException {
        ConfigReader mlpConfig = ConfigReader.getInstance();
        String [] nextLine;
        List<List<Double>> matrix = new ArrayList<List<Double>>();
        while ((nextLine = reader.readNext()) != null) {
            List<Double> exampleValues = new ArrayList<>();
            boolean lineError = false;
            for(String value : nextLine){
                double doubleValue;
                try {
                    doubleValue = Double.parseDouble(value);
                }
                catch(NumberFormatException e)
                {
                    lineError = true;
                    break;
                }
                exampleValues.add(doubleValue);
            }
            if (!lineError) {
                matrix.add(exampleValues);
            }
        }
        return matrix;
    }

    //scale into [-1,1] (min max normanalization)
    public List<List<Double>> normalize(List<List<Double>> matrix)
    {
        List<List<Double>> variables = new ArrayList<>();
        for(int j =0; j < matrix.get(1).size(); j++){
            List<Double> column = new ArrayList<>();
            for(int i =0; i < matrix.size(); i++){
                column.add(matrix.get(i).get(j));
            }
            variables.add(column);
        }

        List<Double> maxOfColums = new ArrayList<>();
        List<Double> minOfColums = new ArrayList<>();
        for(List<Double> column : variables){
            maxOfColums.add(Collections.max(column));
            minOfColums.add(Collections.min(column));
        }

        for(int i =0; i < matrix.size(); i++) {
            for (int j = 0; j < matrix.get(i).size(); j++) {
                Double x = matrix.get(i).get(j);
                Double max = maxOfColums.get(j);
                Double min = minOfColums.get(j);

                Double normalized = 2 * (x - min) / (max - min) -1;
                matrix.get(i).set(j, normalized);
            }
        }
        return matrix;
    }

    public Double[][] getInputsMatrixWithBiasInput(List<List<Double>> dataMatrix){
        ConfigReader  mlpConfig = ConfigReader.getInstance();

        int numberOfInputVariables = mlpConfig.getInputVectors().size();
        Double[][] inputsMatrix = new Double[dataMatrix.size()][numberOfInputVariables + 1];

        int i = 0;
        int k = 0;
        List<Integer> inputVectorsIndices = mlpConfig.getInputVectors();
        for (List<Double> row : dataMatrix) {
            k = 0;
            for(int j : inputVectorsIndices){
                inputsMatrix[i][k] = row.get(j);
                k++;
            }
            inputsMatrix[i][k] = 1.0;
            i++;
        }
        return inputsMatrix;
    }


    public Double[] getOutputVector(List<List<Double>> dataMatrix){
        ConfigReader  mlpConfig = ConfigReader.getInstance();

        int numberOfInptutVaribles = mlpConfig.getInputVectors().size();
        Double[] outputsVector = new Double[dataMatrix.size()];

        int i = 0;
        int outputVectorIndex = mlpConfig.getOutputVector();

        for (List<Double> row : dataMatrix) {
            outputsVector[i] = row.get(outputVectorIndex);
            i++;
        }

        return outputsVector;
    }

    public void splitDataSet(double trainingSetRelativeSize, double validationSetRelativeSize, double testSetRelativeSize, int dataSetSize){

        int trainingLimit = (int)(trainingSetRelativeSize * dataSetSize);
        int validationLimit = (int)((trainingSetRelativeSize + validationSetRelativeSize) * dataSetSize);

        List<Integer> dataSetIndices = new ArrayList<Integer>();

        for (int i=0; i < dataSetSize; i++) {
            dataSetIndices.add(i);
        }
        Collections.shuffle(dataSetIndices);

        trainingSetIndices = new ArrayList<Integer>();
        for(int i = 0; i < trainingLimit; i++){
            trainingSetIndices.add(dataSetIndices.get(i));
        }

        validationSetIndices = new ArrayList<Integer>();
        for(int i = trainingLimit; i < validationLimit; i++){
            validationSetIndices.add(dataSetIndices.get(i));
        }

        testSetIndices = new ArrayList<Integer>();
        for(int i = validationLimit; i < dataSetSize; i++){
            testSetIndices.add(dataSetIndices.get(i));
        }
    }

    List<List<Double>> getTrainingSet(List<List<Double>> dataSet){
        List<List<Double>> trainingSet = new ArrayList<List<Double>>();
        for(Integer index : trainingSetIndices){
            trainingSet.add(dataSet.get(index));
        }
        return trainingSet;
    }

    List<List<Double>> getValidationSet(List<List<Double>> dataSet){
        List<List<Double>> validatioSet = new ArrayList<List<Double>>();
        for(Integer index : validationSetIndices){
            validatioSet.add(dataSet.get(index));
        }
        return validatioSet;
    }

    List<List<Double>> getTestSet(List<List<Double>> dataSet){
        List<List<Double>> testSet = new ArrayList<List<Double>>();
        for(Integer index : testSetIndices){
            testSet.add(dataSet.get(index));
        }
        return testSet;
    }
}