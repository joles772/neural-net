import java.util.ArrayList;

/**
 * Created by Audentity on 3/6/2017.
 */
public class RProp extends BackProp {

    boolean inititalized = false;
    double[][] oldDeltaHL1;
    double[][] oldDeltaOL;

    double etaMax = 1.2;
    double etaMin = 0.5;

    public RProp(double weightsHL1[][], double weightsOL[][], ArrayList<Example> trainingSet, ArrayList<Example> validationSet, double alpha, double eta, Boolean isLogistic){
        this.weightsHL1 = weightsHL1;
        this.weightsOL = weightsOL;
        this.trainingSet = trainingSet;
        this.validationSet = validationSet;
        this.isLogistic = isLogistic;
        this.alpha = alpha;
        this.eta = eta;
    }

    public RProp(){

        super();
    }

    @Override
    public void updateWeights(){

        //System.out.println("Hi");

        if(!inititalized){
            oldDeltaHL1 = new double[weightsHL1.length][weightsHL1[0].length];
            oldDeltaOL = new double[weightsOL.length][weightsOL[0].length];
        }

        for(int i = 0; i < weightsOL.length; i++){
            for(int j = 0; j < weightsOL[0].length; j++){

                if(oldOL[i][j]*errorOL[i][j] > 0){
                    oldDeltaOL[i][j] = Math.min(etaMax*oldDeltaOL[i][j], 50.0);
                    if(errorOL[i][j] < 0) oldDeltaOL[i][j] = oldDeltaOL[i][j]*(-1);
                }
                else if (oldOL[i][j]*errorOL[i][j] < 0){
                    oldDeltaOL[i][j] = etaMin*oldDeltaOL[i][j];
                }
                else{
                    weightsOL[i][j] = weightsOL[i][j];
                }
            }
        }

        for(int i = 0; i < weightsHL1.length; i++){
            for(int j = 0; j < weightsHL1[0].length; j++){

                if(errorHL1[i][j]*oldHL1[i][j] < 0){

                }
                else if(oldHL1[i][j]*errorHL1[i][j] > 0){
                    oldDeltaHL1[i][j] = etaMax*oldDeltaHL1[i][j];
                }
                else if (oldHL1[i][j]*errorHL1[i][j] < 0){
                    oldDeltaHL1[i][j] = etaMin*oldDeltaHL1[i][j];
                }
                else{
                    weightsHL1[i][j] = weightsHL1[i][j];
                }
                if(errorHL1[i][j]*oldHL1[i][j] < 0){
                    weightsHL1[i][j] = weightsHL1[i][j] - oldHL1[i][j];
                    errorHL1[i][j] = 0;
                }
                else if(errorHL1[i][j] > 0){
                    weightsHL1[i][j] = weightsHL1[i][j] - oldDeltaHL1[i][j];
                }
                else if (errorHL1[i][j] < 0){
                    weightsHL1[i][j] = weightsHL1[i][j] + oldDeltaHL1[i][j];
                }
                else{
                    weightsHL1[i][j] = weightsHL1[i][j];
                }
            }
        }
    }
}
