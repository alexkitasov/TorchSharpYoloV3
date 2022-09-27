namespace TorchSharpYoloV3.YoloV3
{
    using TorchSharp;
    using TorchSharp.Utils;
    using static TorchSharp.torch;
    using static TorchSharp.torch.nn;
    using static Utils;
    using System;


    internal partial class Model : Module
    {
        private int num_classes;
        private int num_anchors;
        private bool jump_start = true;
        private Tensor[] samples;
        private Device _device;
        public Model(string path, int num_classes, int num_anchors, int image_size = 416) 
            : this(num_classes,num_anchors)
        {
            this.load(path);
            this.to(_device);
        }
        public Model(int num_classes, int num_anchors_per_scale, int image_size = 416) : base("YoloV3_full")
        {
            this.num_classes = num_classes;
            this.num_anchors = num_anchors_per_scale;
            RegisterComponents();
            _device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;

            Tensor starter = torch.randn(1, this.num_anchors, image_size, image_size);
            this.forward(starter);
            jump_start = false;
        }

        public Device device() {
            return _device;
        }
        public int GetNoOfClassesKnow()
        {
            return num_classes;
        }
        public int GetNoOfAnchPerScale() {
            return num_anchors;
        }
        public int[] GetGrids() {
            List<int> grid = new List<int>();
            foreach (Tensor t in samples) {
                grid.Add((int)t.shape[2]);
            }        
            return grid.ToArray();
        }

    }
}
