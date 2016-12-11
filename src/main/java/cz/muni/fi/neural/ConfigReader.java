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

    public int getInputSize() throws NumberFormatException{
        String prop = props.getProperty("inputSize");
        return Integer.parseInt(prop);
    }

    public int getOutputSize() throws NumberFormatException{
        String prop = props.getProperty("outputSize");
        return Integer.parseInt(prop);
    }

    public int getBatchSize() throws NumberFormatException{
        String prop = props.getProperty("batchSize");
        Integer value = Integer.parseInt(prop);
        return value;
    }

    public boolean initializationDebug()throws NumberFormatException{
       return Boolean.parseBoolean(props.getProperty("initializationDebug"));
    }

    public boolean learningIterationsDebug()throws NumberFormatException{
        return Boolean.parseBoolean(props.getProperty("learningIterationsDebug"));
    }

    public boolean neuronInputsDebug()throws NumberFormatException{
        return Boolean.parseBoolean(props.getProperty("neuronInputsDebug"));
    }

    public boolean outputsOfLearningDebug()throws NumberFormatException{
        return Boolean.parseBoolean(props.getProperty("outputsOfLearningDebug"));
    }

}
