package cz.muni.fi.neural;

import au.com.bytecode.opencsv.CSVReader;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;

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

    public ArrayList<ArrayList<Double>> csvToMatrix() throws IOException {
        String [] nextLine;
        ArrayList<ArrayList<Double>> matrix = new ArrayList<ArrayList<Double>>();
        while ((nextLine = reader.readNext()) != null) {
            ArrayList<Double> exampleValues = new ArrayList<>();
            for(String value : nextLine){
                double doubleValue;
                try {
                    doubleValue = Double.parseDouble(value);
                }
                catch(NumberFormatException e)
                {
                    break;
                }
                exampleValues.add(doubleValue);
            }
            matrix.add(exampleValues);
        }
        return matrix;
    }






}
