import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

/**
 * Created by Audentity on 3/5/2017.
 */
public class TestHarness {

    public TestHarness(){

        //Alter these variables to run the different algorithms with different parameters

        double alpha = 0.8; //The momentum rate
        double eta = 0.2;  //The learning rate
        int sizeHL1 = 32; //The hidden layer size
        boolean isLogistic = true; // true = Logistic activation, false = TanH activation
        int prop = 0; //0 = BackProp, 1 = RProp, 2 = QuickProp
        int type = 0; //0 = Regular, 1 = Holdout, 2 = Cross-Validate

        doNetwork(0, alpha, eta, sizeHL1, isLogistic, prop);

        /*
        //Parameters used for testing the report
        //BP
        doNetwork(0, 0.8, 0.2, 128, true, 0);
        doNetwork(0, 0.8, 0.2, 32, true, 0);
        doNetwork(0, 0.2, 0.8, 128, true, 0);
        doNetwork(0, 0.2, 0.8, 32, true, 0);
        doNetwork(0, 0.5, 0.5, 128, true, 0);
        doNetwork(0, 0.5, 0.5, 32, true, 0);

        doNetwork(0, 0.8, 0.2, 128, false, 0);
        doNetwork(0, 0.8, 0.2, 32, false, 0);

        //RP
        doNetwork(0, 0.8, 0.2, 128, false, 1);
        doNetwork(0, 0.8, 0.2, 32, false, 1);
        doNetwork(0, 0.8, 0.2, 128, true, 1);
        doNetwork(0, 0.8, 0.2, 32, true, 1);

        //QP
        doNetwork(0, 0.8, 0.2, 128, false, 2);
        doNetwork(0, 0.8, 0.2, 32, false, 2);
        doNetwork(0, 0.8, 0.2, 128, true, 2);
        doNetwork(0, 0.8, 0.2, 32, true, 2);

        //Holdout BP
        doNetwork(1, 0.8, 0.2, 128, true, 0);
        //Holdout RP
        doNetwork(1, 0.8, 0.2, 128, true, 1);
        //Holdout QP
        doNetwork(1, 0.8, 0.2, 128, true, 2);

        //Cross-Validate BP
        doNetwork(2, 0.8, 0.2, 128, true, 0);
        //Cross-Validate RP
        doNetwork(2, 0.8, 0.2, 128, true, 1);
        //Cross-Validate QP
        doNetwork(2, 0.8, 0.2, 128, true, 2);
         */

    }

    private void doNetwork(int type, double alpha, double eta, int sizeHL1, boolean isLogistic, int prop){
        String s = "";
        if(prop == 0) s = "BP";
        if(prop == 1) s = "RP";
        if(prop == 2) s = "QP";

        String s2 = "";
        if(isLogistic) s2 = "Log";
        else s2 = "TanH";

        if(type == 0){
            double regErrors = 0;
            double temp = 0;

            System.out.println(s + " " + s2 + " Alpha = " + alpha + " Eta = " + eta +  " HL Size = " + sizeHL1 );
            for(int i = 0; i < 20; i++){
                temp = doRegular(alpha,eta, sizeHL1, isLogistic, prop);
                System.out.println(temp);
                regErrors = regErrors + temp;
            }
            System.out.println("Total Errors: " + regErrors);
            System.out.println("Mean Error: " + regErrors/20);
        }
        else if(type == 1){
            double regErrors = 0;
            double temp = 0;
            System.out.println("Holdout " + s + " " + s2 + " Alpha = " + alpha + " Eta = " + eta +  " HL Size = " + sizeHL1 );
            for(int i = 0; i < 20; i++){
                temp = doHoldout(alpha,eta, sizeHL1, isLogistic, prop, true);
                System.out.println(temp);
                regErrors = regErrors + temp;
            }
            System.out.println("Total Errors: " + regErrors);
            System.out.println("Mean Error: " + regErrors/20);
        }
        else if( type == 2){
            double regErrors = 0;
            double temp = 0;
            System.out.println("Cross=Validation  " + s + " " + s2 + " Alpha = " + alpha + " Eta = " + eta +  " HL Size = " + sizeHL1 );
            for(int i = 0; i < 20; i++){
                temp = crossValidate(alpha,eta, sizeHL1, isLogistic, 2);
                System.out.println(temp);
                regErrors = regErrors + temp;
            }
            System.out.println("Total Errors: " + regErrors);
            System.out.println("Mean Error: " + regErrors/20);
        }
    }

    private int doRegular(double alpha, double eta, int sizeHL1, boolean isLogistic, int prop){
        BackProp n = new RProp();
        if(prop == 0) n = new BackProp();
        else if (prop == 1) n = new RProp();
        else  if(prop == 2) n = new QuickProp();
        else System.exit(0);
        n.alpha = alpha;
        n.eta = eta;
        n.sizeHL1 = sizeHL1;
        n.isLogistic = isLogistic;

        n.makeArrays();

        int epochs = 50;

        for(int i = 0; i < epochs; i++){
            n.train();
        }

        return n.test();
    }

    private int doHoldout(BackProp p, int prop){
        return doHoldout(p.alpha,p.eta,p.sizeHL1, p.isLogistic, prop, false);
    }

    private int doHoldout(double alpha, double eta, int sizeHL1, boolean isLogistic, int prop, boolean print){
        int epochs = 100;
        double currentErrors = 0;
        double oldErrors = Double.MAX_VALUE;

        BackProp n = new BackProp();
        if(prop == 0) n = new BackProp();
        else if (prop == 1) n = new RProp();
        else  if(prop == 2) n = new QuickProp();
        else System.exit(0);
        n.alpha = alpha;
        n.eta = eta;
        n.sizeHL1 = sizeHL1;
        n.isLogistic = isLogistic;

        n.makeArrays();

        double[][] tempWeightsHL1 = new double[1][1];
        double[][] tempWeightsOL = new double[1][1];

        n = holdOut(n);

        for(int i = 0; i < epochs; i++){
            n.train();
            if(i%5 == 0){
                currentErrors = n.validate();
                if(currentErrors > oldErrors) {
                    if(print) System.out.print("Exit Epoch: " + i + ", ");
                    n.weightsOL = tempWeightsOL;
                    n.weightsHL1 = tempWeightsHL1;
                    break;
                }
                else{
                    tempWeightsHL1 = n.weightsHL1;
                    tempWeightsOL = n.weightsOL;
                    oldErrors = currentErrors;
                }
            }

        }
        return n.test();
    }

    private BackProp holdOut(BackProp n){
        BackProp p;
        double errors = 0;

        ArrayList<Example> wholeSet = new ArrayList<>();

        for(int i = 0; i < n.trainingSet.size(); i++){
            wholeSet.add(new Example(n.trainingSet.get(i).data, n.trainingSet.get(i).expected));
        }

        long seed = System.nanoTime();
        Collections.shuffle(wholeSet, new Random(seed));

        ArrayList<Example> trainingSet =  new ArrayList<>(wholeSet.subList(0, (int)(wholeSet.size()*0.3)*2));
        ArrayList<Example> validationSet = new ArrayList<>(wholeSet.subList((int)(wholeSet.size()*0.3)*2 + 1, (wholeSet.size())));
        p = new BackProp(n.weightsHL1, n.weightsOL, trainingSet, validationSet, n.alpha, n.eta, n.isLogistic);
        p.sizeHL1 = n.sizeHL1;
        return p;
    }

    private double crossValidate(double alpha, double eta, int sizeHL1, boolean isLogistic, int prop){
        BackProp p = new BackProp();
        BackProp n = new BackProp();
        if(prop == 0) n = new BackProp();
        else if (prop == 1) n = new RProp();
        else  if(prop == 2) n = new QuickProp();
        else System.exit(0);
        n.alpha = alpha;
        n.eta = eta;
        n.sizeHL1 = sizeHL1;
        n.isLogistic = isLogistic;

        n.makeArrays();

        ArrayList<Example> wholeSet = new ArrayList<>();

        for(int i = 0; i < n.trainingSet.size(); i++){
            wholeSet.add(new Example(n.trainingSet.get(i).data, n.trainingSet.get(i).expected));
        }

        long seed = System.nanoTime();
        Collections.shuffle(wholeSet, new Random(seed));

        ArrayList<Example> partitions[] = new ArrayList[10];

        partitions[0] = new ArrayList<>(wholeSet.subList(0, (int)(wholeSet.size()*0.1)));
        partitions[1] = new ArrayList<>(wholeSet.subList((int)(wholeSet.size()*0.1) + 1, (int)(wholeSet.size()*0.1)*2));
        partitions[2] = new ArrayList<>(wholeSet.subList((int)(wholeSet.size()*0.1)*2 + 1, (int)(wholeSet.size()*0.1)*3));
        partitions[3] = new ArrayList<>(wholeSet.subList((int)(wholeSet.size()*0.1)*3 + 1, (int)(wholeSet.size()*0.1)*4));
        partitions[4] = new ArrayList<>(wholeSet.subList((int)(wholeSet.size()*0.1)*4 + 1, (int)(wholeSet.size()*0.1)*5));
        partitions[5] = new ArrayList<>(wholeSet.subList((int)(wholeSet.size()*0.1)*5 + 1, (int)(wholeSet.size()*0.1)*6));
        partitions[6] = new ArrayList<>(wholeSet.subList((int)(wholeSet.size()*0.1)*6 + 1, (int)(wholeSet.size()*0.1)*7));
        partitions[7] = new ArrayList<>(wholeSet.subList((int)(wholeSet.size()*0.1)*7 + 1, (int)(wholeSet.size()*0.1)*8));
        partitions[8] = new ArrayList<>(wholeSet.subList((int)(wholeSet.size()*0.1)*8 + 1, (int)(wholeSet.size()*0.1)*9));
        partitions[9] = new ArrayList<>(wholeSet.subList((int)(wholeSet.size()*0.1)*9 + 1, wholeSet.size()));

        ArrayList<Example> trainingSet;
        ArrayList<Example> validationSet;

        BackProp nets[] = new BackProp[10];

        for(int k = 0; k < 10; k++){

            trainingSet = new ArrayList<>();
            validationSet = new ArrayList<>();

            for(int i = 0; i < 10; i++){
                //System.out.println(i);
                if(i != k) trainingSet.addAll(partitions[i]);
            }
            validationSet.addAll(partitions[k]);

            if(prop == 0) p = new BackProp(n.weightsHL1, n.weightsOL, trainingSet, validationSet, n.alpha, n.eta, n.isLogistic);
            else if (prop == 1) p = new RProp(n.weightsHL1, n.weightsOL, trainingSet, validationSet, n.alpha, n.eta, n.isLogistic);
            else  if(prop == 2) p = new QuickProp(n.weightsHL1, n.weightsOL, trainingSet, validationSet, n.alpha, n.eta, n.isLogistic);

            p.sizeHL1 = n.sizeHL1;
            nets[k] = p;
        }

        int min = Integer.MAX_VALUE;
        int temp;

        for(BackProp b: nets){
            temp = doHoldout(b, prop);
            if(temp < min) min = temp;
        }
        return min;
    }

    public static void main(String[] args) {
        TestHarness t = new TestHarness();
    }
}
