
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Locale;
import java.util.Scanner;

/**
 * Created by Audentity on 2/26/2017.
 */
public class InputReader {
    int expected;
    String path;
    Scanner sc;

    public ArrayList<Double[]> data;

    public InputReader(String path, int expected) throws FileNotFoundException {
        this.path = path;
        this.expected = expected;

        Double[] out;
        sc = new Scanner(new File(path));
        sc.useLocale(Locale.ENGLISH);
        data = new ArrayList<>();


        while (sc.hasNextLine()) {

            String s = sc.nextLine();
            String[] ar = s.split(",");
            out = new Double[ar.length];
            for(int i = 0; i < ar.length; i++){
                out[i] = Double.parseDouble(ar[i]);
                //System.out.println(out[i]);
            }
            data.add(out);
        }
    }
}
