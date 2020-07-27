using System;
using System.IO;
using System.Linq;
using TensorFlow;


namespace TensorFlowNew
{


    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            /*
            using (var session = new TFSession())
            {
                var graph = session.Graph;

                var a = graph.Const(2);
                var b = graph.Const(3);
                Console.WriteLine("a=2 b=3");

                // Add two constants
                var addingResults = session.GetRunner().Run(graph.Add(a, b));
                var addingResultValue = addingResults.GetValue();
                Console.WriteLine("a+b={0}", addingResultValue);

                // Multiply two constants
                var multiplyResults = session.GetRunner().Run(graph.Mul(a, b));
                var multiplyResultValue = multiplyResults.GetValue();
                Console.WriteLine("a*b={0}", multiplyResultValue);
            }
			*/

            //XorTensorCalculate();
            //ManCalcXorUsingMatrix();

            XorNeuroTry();
        }

        public static void ManCalcXorUsingMatrix()
        {
            Console.WriteLine("Basic matrix");
            using (var g = new TFGraph())
            {
                var s = new TFSession(g);

                // 1x4 matrix
                var matrix1 = g.Const(new int[,] { { 0, 0, 1, 1  } });
                // 1x4 matrix
                var matrix2 = g.Const(new int[,] { { 0 , 1, 0, 1 } });

                // xor
                var xor = g.BitwiseXor(matrix1, matrix2);


                var result = s.GetRunner().Run(xor);
                Console.WriteLine("Tensor ToString=" + result);

                var resultAsArray = (int[,])result.GetValue();


                Console.WriteLine("Value =");
                foreach(var res in resultAsArray)
                {
                    Console.WriteLine(res);
                }

            };
        }

        public static void XorNeuroTry()
        {
            var trainData = new double[] {0.0, 1.0};

            var n_samples = trainData.Length;

            
            using (var g = new TFGraph())
            {
                var s = new TFSession(g);
                var rng = new Random(0);
                // tf Graph Input

               
                var X1 = g.Placeholder(TFDataType.Double);
                var X2 = g.Placeholder(TFDataType.Double);

                //расчетов начальных весов
                var W = g.Variable(g.Const(rng.NextDouble()), operName: "weight");

                //не уверен, что рассчет смещения рандомным образом - хорошая идея.
                var b = g.Variable(g.Const(rng.NextDouble()), operName: "bias");

                //вход умноженный на весовой коэффициент плюс смещение = операция которая вычисляет взвешенную сумма весов.
                var predX1 = g.Add(g.Mul(X1, W.Read, "x1_w"), b.Read);
                var predX2 = g.Add(g.Mul(X2, W.Read, "x2_w"), b.Read);

                var pred = g.Add(predX1, predX2);

                var cost = g.Sigmoid(pred);

                var learning_rate = 0.001f;
                var training_epochs = 100;
                var sgd = new SGD(g, learning_rate);
                var updateOps = sgd.Minimize(cost);

                using (var sesssion = new TFSession(g))
                {
                    sesssion.GetRunner().AddTarget(g.GetGlobalVariablesInitializer()).Run();

                    for (int i = 0; i < training_epochs; i++)
                    {
                        double avgLoss = 0;
                        for (int j = 0; j < n_samples; j++)
                        {
                            var tensors = sesssion.GetRunner()
                               .AddInput(X1, new TFTensor(trainData[j]))
                               .AddInput(X2, new TFTensor(trainData[j]))
                               .AddTarget(updateOps).Fetch(sgd.Iterations.Read, cost, W.Read, b.Read, sgd.LearningRate).Run();
                            avgLoss += (double)tensors[1].GetValue();
                        }
                        var tensors2 = sesssion.GetRunner()
                               .Fetch(W.Read, b.Read).Run();
                        var output = $"Epoch: {i + 1:D}, loss: {avgLoss / n_samples:F4}, W: {tensors2[0].GetValue():F4}, b: {tensors2[1].GetValue():F4}";
                        Console.WriteLine(output);
                    }
                }
            }
            
        }

        public static void test2()
        {
            


        }


            public static void test1()
        {
            
                Console.WriteLine("Linear regression");
                // Parameters
                var learning_rate = 0.001f;
                var training_epochs = 100;
                var display_step = 50;

                // Training data
                var train_x = new double[] {
                3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1
            };
                var train_y = new double[] {
                1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                 2.827,3.465,1.65,2.904,2.42,2.94,1.3
            };
                var n_samples = train_x.Length;
                using (var g = new TFGraph())
                {
                    var s = new TFSession(g);
                    var rng = new Random(0);
                    // tf Graph Input
                    
                    var X = g.Placeholder(TFDataType.Double);
                    var Y = g.Placeholder(TFDataType.Double);

                    var W = g.Variable(g.Const(rng.NextDouble()), operName: "weight");
                    var b = g.Variable(g.Const(rng.NextDouble()), operName: "bias");
                    var pred = g.Add(g.Mul(X, W.Read, "x_w"), b.Read);

                    var cost = g.Div(g.ReduceSum(g.Pow(g.Sub(pred, Y), g.Const(2.0))), g.Mul(g.Const(2.0), g.Const((double)n_samples), "2_n_samples"));

                    // SOLVED
                    // STuck here: TensorFlow bindings need to surface gradient support
                    // waiting on Google for this
                    // https://github.com/migueldeicaza/TensorFlowSharp/issues/25

                    var sgd = new SGD(g, learning_rate);
                    var updateOps = sgd.Minimize(cost);

                    using (var sesssion = new TFSession(g))
                    {
                        sesssion.GetRunner().AddTarget(g.GetGlobalVariablesInitializer()).Run();

                        for (int i = 0; i < training_epochs; i++)
                        {
                            double avgLoss = 0;
                            for (int j = 0; j < n_samples; j++)
                            {
                                var tensors = sesssion.GetRunner()
                                   .AddInput(X, new TFTensor(train_x[j]))
                                   .AddInput(Y, new TFTensor(train_y[j]))
                                   .AddTarget(updateOps).Fetch(sgd.Iterations.Read, cost, W.Read, b.Read, sgd.LearningRate).Run();
                                avgLoss += (double)tensors[1].GetValue();
                            }
                            var tensors2 = sesssion.GetRunner()
                                   .Fetch(W.Read, b.Read).Run();
                            var output = $"Epoch: {i + 1:D}, loss: {avgLoss / n_samples:F4}, W: {tensors2[0].GetValue():F4}, b: {tensors2[1].GetValue():F4}";
                            Console.WriteLine(output);
                        }
                    }
                }
            
        }

        #region XorManualCalc
        public static void XorTensorCalculate()
        {
            XorTensorCalculation(0, 0);
            XorTensorCalculation(0, 1);
            XorTensorCalculation(1, 0);
            XorTensorCalculation(1, 1);
        }

        public static void XorTensorCalculation(int x1c, int x2c)
        {
            using (var g = new TFGraph())
            {
                var s = new TFSession(g);

                var x1 = g.Placeholder(TFDataType.UInt8);
                var x2 = g.Placeholder(TFDataType.UInt8);

                var runner = s.GetRunner();
                runner.AddInput(x1, (byte)x1c);
                runner.AddInput(x2, (byte)x2c);

                var xor = g.BitwiseXor(x1, x2);

                Console.WriteLine("x1^x2={0}", runner.Run(xor).GetValue());
            }
        }

        #endregion XorManualCalc
    }
}
