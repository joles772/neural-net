import java.util.ArrayList;

/**
 * Created by Audentity on 3/6/2017.
 */
public class QuickProp extends BackProp {

    boolean inititalized = false;
    double[][] oldDeltaHL1;
    double[][] oldDeltaOL;

    public QuickProp(double weightsHL1[][], double weightsOL[][], ArrayList<Example> trainingSet, ArrayList<Example> validationSet, double alpha, double eta, Boolean isLogistic){
        this.weightsHL1 = weightsHL1;
        this.weightsOL = weightsOL;
        this.trainingSet = trainingSet;
        this.validationSet = validationSet;
        this.isLogistic = isLogistic;
        this.alpha = alpha;
        this.eta = eta;
    }

    public QuickProp(){

        super();
    }

    @Override
    public void updateWeights(){



        if(!inititalized){
            oldDeltaHL1 = new double[weightsHL1.length][weightsHL1[0].length];
            oldDeltaOL = new double[weightsOL.length][weightsOL[0].length];

            for(int i = 0; i < weightsOL.length; i++){
                for(int j = 0; j < weightsOL[0].length; j++){
                    oldDeltaOL[i][j] = - (eta*errorOL[i][j]);
                    weightsOL[i][j] = weightsOL[i][j] +oldDeltaOL[i][j];
                }
            }

            for(int i = 0; i < weightsHL1.length; i++){
                for(int j = 0; j < weightsHL1[0].length; j++){
                    oldDeltaHL1[i][j] = - (eta*errorHL1[i][j]);
                   weightsHL1[i][j] = weightsHL1[i][j] + oldDeltaHL1[i][j];
                }
            }

            inititalized = false;
         }
         else{
            for(int i = 0; i < weightsOL.length; i++){
                for(int j = 0; j < weightsOL[0].length; j++){
                    oldDeltaOL[i][j] = oldDeltaOL[i][j]*(errorOL[i][j]/(-errorOL[i][j] + oldOL[i][j]));
                    weightsOL[i][j] = weightsOL[i][j] + oldDeltaOL[i][j];
                }
            }

            for(int i = 0; i < weightsHL1.length; i++){
                for(int j = 0; j < weightsHL1[0].length; j++){
                    oldDeltaHL1[i][j] = oldDeltaHL1[i][j]*(errorHL1[i][j]/(-errorHL1[i][j] + oldHL1[i][j]));
                    weightsHL1[i][j] = weightsHL1[i][j] + oldDeltaHL1[i][j];
                }
            }
        }
    }
}
