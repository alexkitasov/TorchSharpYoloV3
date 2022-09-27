namespace TorchSharpYoloV3
{
    using TorchSharp;
    using static TorchSharp.torch;
    using static TorchSharp.torch.nn;
    using static TorchSharp.torch.utils.data;

    using YoloV3;
    using static Utils;
    using System.Linq;
    using System.Collections;
    using System.Drawing;
    using System.Threading.Channels;
    using System.Data;
    using static TorchSharp.torch.optim;
    using System.Drawing.Drawing2D;

    public class ModelHandler
    {
        private Model _model;
        private static int _num_classes;
        private static int _num_anchors;
        private static int _channels;
        private static int _image_size;
        private static int _max_objects = 50;
        private static string _root_path = "";
        private static string _dataset_name = "";
        public ModelHandler(int num_classes, int num_anchors, int image_size, int channels = 3)
        {
            _num_classes = num_classes;
            _num_anchors = num_anchors;
            _image_size = image_size;
            _channels = channels;
            _model = new Model(_num_classes, _num_anchors, _image_size);
            _model.to(_model.device());
        }
        public void SaveModel(string path)
        {
            _model.save(path);
        }

        public void CreateNewloadedModel(string path)
        {
            _model = new Model(path, _num_classes, _num_anchors, _image_size);
        }
        public Module model() {
            return _model;
        }
        public DataLoader data_loader(string root, string dataset_name, int batch_size, bool shuffle = false)
        {
            _root_path = root;
            _dataset_name = dataset_name;

            ModelDataset ds = dataset(root, dataset_name);
            return new DataLoader(
                dataset: ds,
                batchSize: batch_size,
                shuffle: shuffle,
                device: _model.device());
        }
        private static ModelDataset dataset(string root, string dataset_name) {
            return new ModelDataset(
                root
                , dataset_name
                , _image_size
                , _channels
                , _max_objects);
        }

        public void train(int epochs, DataLoader data, float[] anchors) { 
            _model.train();
            Tensor all_anchors = anchors;
            if (all_anchors.shape[0] < 2)
                throw new Exception("invalid number of anchors");
            all_anchors = all_anchors.reshape(-1, 2);

            Tensor[] scaled_anchor
                = GetScaledAnchors(
                    anchors: all_anchors,
                    scales: _model.GetGrids(),
                    device: _model.device());

            float l_r = 0.001f;
            float w_d = 0.0005f;
            float momentum = 0.9f;
            float nms_iou_threshold = 0.45f;
            float map_iou_threshold = 0.5f;
            float conf_threshold = 0.05f;

            var optimizer = torch.optim.Adam(_model.parameters(), lr: l_r, weight_decay: w_d);
            var scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, 0.95);
            var iterater = data.GetEnumerator();
            int runs = data.Count();
            foreach (int epoch in Enumerable.Range(0, epochs))
            {
                foreach (int batch_id in Enumerable.Range(0, runs)) {
                    iterater.MoveNext();
                    var x = iterater.Current["data"];
                    var y = iterater.Current["label"];
                    var batch_size = (int)(x.shape[0]);

                    using (var d = torch.NewDisposeScope())
                    {
                        optimizer.zero_grad();
                        // ========================================
                        // Predict
                        // ========================================
                        Tensor output = _model.forward(x);
                        // unpack prediction
                        Tensor[] predictions
                            = GetUnpackPredictions(
                                output: output,
                                scaled_anchor: scaled_anchor,
                                num_scales: _model.GetGrids().Length,
                                batch_size: batch_size,
                                num_anchors: _num_anchors,
                                grid_size: _model.GetGrids(),
                                num_classes: _num_classes);
                        // construct target per scale
                        Tensor[] target
                            = GetTransformedTargets(
                                target: y,
                                unscaled_anchors: all_anchors,
                                scales: _model.GetGrids(),
                                device: _model.device());
                        // ========================================
                        // Caltulate Loss
                        // ========================================
                        // calculate first scale to initialize losses
                        var losses = ModelLoss.GetLossPerScale(
                            predictions[0],
                            target[0],
                            scaled_anchor[0]);
                        // calculate the rest of scales losses
                        foreach (int i in Enumerable.Range(1, _model.GetGrids().Length - 1))
                        {
                            losses = losses + ModelLoss.GetLossPerScale(
                                predictions[i],
                                target[i],
                                scaled_anchor[i]);
                        }
                        // ========================================
                        // Run Optimization
                        // ========================================
                        losses.backward();
                        optimizer.step();
                        // ========================================
                        // Report intermediate results
                        // and maybe save model
                        // ========================================
                        if (batch_id == runs - 1)
                        {
                            if ((epoch % 1) == 0)
                            {                                
                                Console.WriteLine($"epoch: {epoch}: tensors: {torch.Tensor.TotalCount} loss: {(float)losses}");
                            }
                        }
                    }
                }
                scheduler.step();
                iterater.Reset();
            }

        }






    }
}
