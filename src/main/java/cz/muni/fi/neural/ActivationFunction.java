package cz.muni.fi.neural;

/**
 * Created by Simon on 06.11.2016.
 */
public interface ActivationFunction {

    double computeOutput(double innerPotential);
    double derivationOutput(double y);
    //getRangeOfValues()
}
