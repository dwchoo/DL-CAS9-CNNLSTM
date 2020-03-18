class model_results:
    def __init__(self):
        self.train_total_loss           = None
        self.train_class_1_loss         = None
        self.train_class_2_loss         = None
        self.train_class_final_loss     = None
        self.train_rate_loss            = None

        self.val_total_loss             = None
        self.val_class_1_loss           = None
        self.val_class_2_loss           = None
        self.val_class_final_loss       = None
        self.val_rate_loss              = None

        self.test_total_loss            = None
        self.test_class_1_loss          = None
        self.test_class_2_loss          = None
        self.test_class_final_loss      = None
        self.test_rate_loss             = None

        self.bench_total_loss           = None
        self.bench_class_1_loss         = None
        self.bench_class_2_loss         = None
        self.bench_class_final_loss     = None
        self.bench_rate_loss            = None


    def correlation_init(self,
                        test_pearson,
                        test_spearman,
                        bench_pearson,
                        bench_spearman):

        self.test_pearson       = test_pearson
        self.test_spearman      = test_spearman
        
        self.bench_pearson        = bench_pearson
        self.bench_spearman       = bench_spearman
        

