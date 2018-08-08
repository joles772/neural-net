import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

/**
 * Created by Audentity on 2/25/2017.
 */
public class BackProp {

    int sizeIL = 8*8;
    int sizeHL1 = 8*8 + 4;

    double alpha = 0.8;
    double eta = 0.2;
    boolean isLogistic = true;

    double weightsHL1[][];//
    double weightsOL[][];//

    double inputL[];
    double outputHL1[];//
    double outputOL[];//

    double errorHL1[][];//
    double errorOL[][];//

    double oldHL1[][];//
    double oldOL[][];//

    InputReader reader;

    ArrayList<Example> trainingSet = new ArrayList<>();
    ArrayList<Example> validationSet = new ArrayList<>();

    Random rand = new Random();

    public BackProp(double weightsHL1[][], double weightsOL[][], ArrayList<Example> trainingSet, ArrayList<Example> validationSet, double alpha, double eta, Boolean isLogistic){
        this.weightsHL1 = weightsHL1;
        this.weightsOL = weightsOL;
        this.trainingSet = trainingSet;
        this.validationSet = validationSet;
        this.isLogistic = isLogistic;
        this.alpha = alpha;
        this.eta = eta;
    }

    public BackProp(){

        try {
            for(int j = 0; j < 10; j++){
                reader = new InputReader("data/digit_train_" + j + ".txt", j);

                for(int i = 0; i < reader.data.size(); i++){
                    trainingSet.add(new Example(reader.data.get(i), reader.expected));
                }
            }
        }
        catch(Exception e){
            System.exit(0);
        }
    }


    public int test(){
        int[] errors = new int[] {0,0,0,0,0,0,0,0,0,0};
        try {
            for(int j = 0; j < 10; j++){
                reader = new InputReader("data/digit_test_" + j + ".txt", j);
                for(int i = 0; i < reader.data.size(); i++){
                    for(int k = 0; k < reader.data.get(i).length; k++){
                        inputL[k] = reader.data.get(i)[k];
                    }
                    forwardPass();
                    if(getResult() != j) errors[j]++;
                }
            }

            int temp = 0;

            for(int i = 0; i < 10; i++){
                temp = temp + errors[i];
            }
            return temp;
        }
        catch(Exception e){
            System.out.println("EXIT");
            System.exit(0);
        }
        return -1;
    }

    private int getResult(){
        int maxNum = -1;
        double maxVal = Double.MIN_VALUE;

        for(int i = 0; i < outputOL.length; i++){
            if(outputOL[i] > maxVal){
                maxVal = outputOL[i];
                maxNum = i;
            }
        }
        return maxNum;
    }

    public int validate(){
        int errors = 0;
        for(int i = 0; i < validationSet.size(); i++){
            inputL = new double[64];
            for(int j = 0; j < validationSet.get(i).data.length; j++){
                inputL[j] = validationSet.get(i).data[j];
            }
            forwardPass();
            if(getResult() != validationSet.get(i).expected) errors++;
        }
        return errors;
    }

    public void train(){

        oldHL1 = new double[weightsHL1.length][weightsHL1[0].length];
        oldOL = new double[weightsOL.length][weightsOL[0].length];

        long seed = System.nanoTime();
        Collections.shuffle(trainingSet, new Random(seed));

        inputL = new double[64];

        for(int i = 0; i < trainingSet.size(); i++){
            for(int j = 0; j < inputL.length; j++){
                inputL[j] = trainingSet.get(i).data[j];
            }
            forwardPass();
            backPropogate(trainingSet.get(i).expected);
        }
    }

    private void forwardPass(){

        outputHL1 = new double[sizeHL1];
        outputOL = new double[10];

        double sum = 0;
        for(int i = 0; i < outputHL1.length; i++){
            sum = 0;

            for(int j = 0; j < weightsHL1[i].length; j++){
                sum = sum + inputL[j]*weightsHL1[i][j];
            }

            if(isLogistic) outputHL1[i] = logistic(sum);
            else outputHL1[i] = tanH(sum);
        }
        for(int i = 0; i < 10; i++){
            sum = 0;

            for(int j = 0; j < weightsOL[i].length; j++){
                sum = sum + outputHL1[j]*weightsOL[i][j];
            }

           outputOL[i] = logistic(sum);
        }

    }

    private void backPropogate(int target){
        errorOL = new double[weightsOL.length][weightsOL[0].length];
        errorHL1 = new double[weightsHL1.length][weightsHL1[0].length];

        for(int i = 0; i < errorOL.length; i++){
            for(int j = 0; j < errorOL[0].length; j++){
                if(i == target) errorOL[i][j] = (outputOL[i] - 1)*outputHL1[j];
                else errorOL[i][j] = (outputOL[i] - 0)*outputHL1[j];
            }
        }

        double sum = 0;

        for(int j = 0; j < errorHL1.length; j++){ //size HL1

            for(int k = 0; k < errorHL1[0].length; k++){ //size IL

                sum = 0;

                for(int i = 0; i < outputOL.length; i++){ //size OL

                    if(isLogistic){
                        if(i == target) sum = sum + (outputOL[i] - 1)*(weightsOL[i][j])*((outputHL1[j])*(1 - outputHL1[j]))*inputL[k];
                        else sum = sum + (outputOL[i] - 0)*(weightsOL[i][j])*((outputHL1[j])*(1 - outputHL1[j]))*inputL[k];
                    }
                    else{
                        if(i == target) sum = sum + (outputOL[i] - 1)*(weightsOL[i][j])*((1 - outputHL1[j]*outputHL1[j]))*inputL[k];
                        else sum = sum + (outputOL[i] + 0)*(weightsOL[i][j])*((1 - outputHL1[j]*outputHL1[j]))*inputL[k];
                    }
                }

                errorHL1[j][k] = sum;
            }
        }

        updateWeights();

        oldOL = errorOL;
        oldHL1 = errorHL1;
    }

    public void updateWeights(){
        for(int i = 0; i < weightsOL.length; i++){
            for(int j = 0; j < weightsOL[0].length; j++){
                weightsOL[i][j] = weightsOL[i][j] - (eta*errorOL[i][j]) - alpha*eta*oldOL[i][j];
            }
        }

        for(int i = 0; i < weightsHL1.length; i++){
            for(int j = 0; j < weightsHL1[0].length; j++){
                weightsHL1[i][j] = weightsHL1[i][j] - (eta*errorHL1[i][j]) - alpha*eta*oldHL1[i][j];
            }
        }
    }

    public void makeArrays(){
        weightsHL1 = new double[sizeHL1][sizeIL];

        for(int i = 0; i < sizeHL1; i ++){
            for(int j = 0; j < sizeIL; j++){
                weightsHL1[i][j] = rand.nextDouble();
            }
        }



        weightsOL = new double[10][sizeHL1];

        for(int i = 0; i < 10; i ++){
            for(int j = 0; j < sizeHL1; j++){
                weightsOL[i][j] = rand.nextDouble()*(rand.nextInt(1 - (-1) + 1) + (-1));
            }
        }

        for(int i = 0; i < sizeHL1; i ++){
            for(int j = 0; j < 64; j++){
                weightsHL1[i][j] = rand.nextDouble()*(rand.nextInt(1 - (-1) + 1) + (-1));
            }
        }

        oldHL1 = new double[weightsHL1.length][weightsHL1[0].length];
        oldOL = new double[weightsOL.length][weightsOL[0].length];
    }

    private double logistic(double z){
        return 1/(1 + Math.pow(Math.E, -z));
    }

    private double tanH(double z){
        return (Math.pow(Math.E, z) - Math.pow(Math.E, -z))/(Math.pow(Math.E, z) + Math.pow(Math.E, -z));
    }
}
