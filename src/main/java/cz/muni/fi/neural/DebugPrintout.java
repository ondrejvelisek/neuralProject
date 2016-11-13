package cz.muni.fi.neural;

import java.io.*;

/**
 * Created by Simon on 13.11.2016.
 */
public class DebugPrintout {
    private static DebugPrintout ourInstance = new DebugPrintout();

    private Writer debug;

    public static DebugPrintout getInstance() {
        return ourInstance;
    }

    private DebugPrintout() {
        try {
            debug = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream("debugPrinOut.txt"), "utf-8"));
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    public void print(String text){
        try {
            debug.write(text + "\n");
            debug.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
