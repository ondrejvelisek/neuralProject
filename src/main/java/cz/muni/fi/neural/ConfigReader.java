package cz.muni.fi.neural;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

/**
 * Created by Simon on 13.11.2016.
 */
public class ConfigReader {

    private InputStream configFile;
    private Properties props;


    private static ConfigReader ourInstance = new ConfigReader();

    public static ConfigReader getInstance() {
        return ourInstance;
    }

    public ConfigReader(){
        try {
            configFile = new FileInputStream("src/main/resources/config.properties");
            props = new Properties();
            props.load(configFile);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public List<Integer> getMlpArchitecture() throws NumberFormatException{
        String prop = props.getProperty("mlpArchitecture");
        String [] propValues = prop.split(",");

        List<Integer> mlpArchitecture = new ArrayList<>();
        for(int i=0; i < propValues.length; i++){
            mlpArchitecture.add(Integer.parseInt(propValues[i]));
        }
        return mlpArchitecture;
    }

    public List<Integer> getInputVectors() throws NumberFormatException{
        String prop = props.getProperty("inputVectors");
        String [] propValues = prop.split(",");

        List<Integer> inputVectors = new ArrayList<>();
        for (int i = 0; i < propValues.length; i++) {
            inputVectors.add(Integer.parseInt(propValues[i]) - 1);
        }
        return inputVectors;
    }

    public int getOutputVector() throws NumberFormatException{
        String prop = props.getProperty("outputVector");
        Integer value = Integer.parseInt(prop) - 1;
        return value;
    }

    public int getBatchSize() throws NumberFormatException{
        String prop = props.getProperty("batchSize");
        Integer value = Integer.parseInt(prop);
        return value;
    }

    public String getDataSourceName(){
        return props.getProperty("dataSourceName");
    }

    public char getCsvSeparator(){
        return props.getProperty("csvSeparator").charAt(0);
    }

    public boolean initializationDebug()throws NumberFormatException{
        return Boolean.parseBoolean(props.getProperty("initializationDebug"));
    }

    public boolean learningError()throws NumberFormatException{
        return Boolean.parseBoolean(props.getProperty("learningError"));
    }

    public boolean validationError()throws NumberFormatException{
        return Boolean.parseBoolean(props.getProperty("validationError"));
    }

    public boolean neuronInputsDebug()throws NumberFormatException{
        return Boolean.parseBoolean(props.getProperty("neuronInputsDebug"));
    }

    public boolean outputsOfLearningDebug()throws NumberFormatException{
        return Boolean.parseBoolean(props.getProperty("outputsOfLearningDebug"));
    }

    public boolean loadedDatasetDebug()throws NumberFormatException{
        return Boolean.parseBoolean(props.getProperty("loadedDatasetDebug"));
    }
}