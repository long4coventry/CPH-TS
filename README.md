# CPH-TS

The dataloader and dataloader1 will be used for data preparation in the stage of retraining and re-sampling respectively. <br>
CPH is the main data imputation algorithm <br>
CPH1 is created to facilitate the query by committee sampling, where we need to find out the cells that have the biggest change with two same networks and thus has the biggest disagreement in its prediction. <br>
The main module is BBI.py, which will perform the entire pipeline.
