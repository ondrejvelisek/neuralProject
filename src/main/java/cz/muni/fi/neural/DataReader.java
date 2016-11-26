package cz.muni.fi.neural;

import au.com.bytecode.opencsv.CSVReader;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Created by Simon on 06.11.2016.
 */
public class DataReader {
    private CSVReader reader;

    public DataReader(){

    }
    public void setFile(String fileName) throws FileNotFoundException {
        reader = new CSVReader(new FileReader(fileName));
    }

    public List<List<Double>> csvToMatrix() throws IOException {
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
        return normalize(matrix);
    }

    //scale into [-1,1] (min max normanalization)
    private List<List<Double>> normalize(List<List<Double>> matrix)
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
}
