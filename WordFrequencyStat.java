package wordCount;

import java.io.*;


import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multiset;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.ClusterClassifier;
import org.apache.mahout.clustering.iterator.KMeansClusteringPolicy;
import org.apache.mahout.clustering.kmeans.Kluster;
import org.apache.mahout.common.distance.CosineDistanceMeasure;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.encoders.FeatureVectorEncoder;
import org.apache.mahout.vectorizer.encoders.StaticWordValueEncoder;
import org.wltea.analyzer.lucene.IKAnalyzer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class WordFrequencyStat {

    public static List<NamedVector> loadData() throws IOException {
        HashMap<String, String> bookMap = new HashMap<>();
        DataInputStream inFile = new DataInputStream(new FileInputStream(new File("F:\\mahout\\day2\\data\\weibo_user.csv")));
        String inString = null;
        BufferedReader reader = new BufferedReader(new InputStreamReader(inFile,"GBK"));
        try {
            while((inString = reader.readLine())!= null ){
                String [] item = inString.split(",",-1);
                String key = item[0];
                String value = item[2];
                System.out.println(item[2]);
                bookMap.put(key,value);
            }
        }catch (FileNotFoundException e) {
            e.printStackTrace();
        }catch (IOException e) {
            e.printStackTrace();
        }finally {
            try {
                reader.close();
            }catch (IOException e) {
                e.printStackTrace();
            }
        }
        // 1，使用IKAnalyzer中文分词器，对于书籍的简介进行分词
        HashMap<String, Multiset<String>> tokenMap = new HashMap<>();
        for (Map.Entry<String, String> entry : bookMap.entrySet()) {
            Multiset<String> words = ConcurrentHashMultiset.create();
            Analyzer analyzer = new IKAnalyzer(true);
            TokenStream ts = analyzer.tokenStream("body", entry.getValue());
            ts.reset();
            while (ts.incrementToken()) {
                String word = ts.getAttribute(CharTermAttribute.class).toString();
                // 1.1 去掉长度为1的字符， 因为 我 的  得 地 这种词对文档聚类不仅不起到作用
                //     反而会影响正确性
                if (word.length() <= 1 || word.equals("简介")) {
                    continue;
                }
                words.add(word);
            }
            tokenMap.put(entry.getKey(), words);
        }
        // 2，使用FeatureVectorEncoder编码把中文分词，转换为Vector
        FeatureVectorEncoder encoder = new StaticWordValueEncoder("body");
        // 3，使用NamedVector做外层包裹，这样让输出结果时，方便知道是哪一本书
        List<NamedVector> bookVectors = new ArrayList<>();
        for (Map.Entry<String, Multiset<String>> entry : tokenMap.entrySet()) {
            // 3.1 因为中文分词，分出来的值不是连续的，使用RandomAccessSparseVector最合适
            //     这里设置为1000，只有1000个分词起作用
            Vector bookVector = new RandomAccessSparseVector(1000);
            Multiset<String> words = entry.getValue();
            for (String word : words.elementSet()) {
                // 3.2 使用TF-IDF计算出每个词的特征值
                encoder.addToVector(word, calcTFIDF(word, words, tokenMap), bookVector);
            }
            NamedVector bookNameVector = new NamedVector(bookVector, entry.getKey());
            bookVectors.add(bookNameVector);
        }
        return bookVectors;
    }

    public static void main(String[] args) throws IOException {
        // 1, 加载数据
        List<NamedVector> bookVectors = loadData();
        List<Cluster> clusterModels = new ArrayList<>();
        for (int i = 0; i < 10; ++i) {
            Vector vec = bookVectors.get(i);
            // 这里使用余弦计算距离
            clusterModels.add(new Kluster(vec, i, new CosineDistanceMeasure()));
        }
        // 3，构建聚类算法
        KMeansClusteringPolicy kMeansClusteringPolicy = new KMeansClusteringPolicy();
        ClusterClassifier classifier = new ClusterClassifier(clusterModels, kMeansClusteringPolicy);
        kMeansClusteringPolicy.update(classifier);
        // 4， 运行算法，这里默认迭代10次
        for (int i = 0; i < 100; ++i) {
            for (NamedVector vector : bookVectors) {
                Vector probabilities = classifier.classify(vector);
                Vector weights = kMeansClusteringPolicy.select(probabilities);
                for (Vector.Element e : weights.nonZeroes()) {
                    int index = e.index();
                    classifier.train(index, vector, weights.get(index));
                }
            }
            // 4.1 每次簇中心迭代完毕，计算先选出来簇中心的变化大小
            classifier.close();
            classifier.getPolicy().update(classifier);
            List<Cluster> models = classifier.getModels();
            boolean bBreak = false;
            // 4.2 当簇中心的变化不大时，推出计算
            for (Cluster cluster : models) {
                if (cluster.isConverged()) {
                    bBreak = true;
                    break;
                }
            }
            if (bBreak) {
                break;
            }
        }
        // 5，根据计算出来的簇中心，计算各个点到各个簇中心的距离，然后和离自己最新的簇中心
        //    成为一簇
        for (NamedVector vector : bookVectors) {
            Vector pdfPerCluster = classifier.classify(vector);
            if (pdfPerCluster.maxValue() >= 0) {
                System.out.printf("name = %s clusterId = %d value = %f\n", vector.getName(), pdfPerCluster.maxValueIndex(), pdfPerCluster.maxValue());
            }
        }


    }


    // 优化
//        // 4，目前应知道有Java， Hadoop，C++， 量化 这四个簇，
//        //    为了提高准确性，直接选中以Java，Hadoop，C++，量化其中的一本书籍作为初始簇中心
//        clusterModels.add(new Kluster(bookVectors.get(0), 0, new CosineDistanceMeasure()));
//        clusterModels.add(new Kluster(bookVectors.get(1), 1, new CosineDistanceMeasure()));
//        clusterModels.add(new Kluster(bookVectors.get(3), 2, new CosineDistanceMeasure()));
//        clusterModels.add(new Kluster(bookVectors.get(7), 3, new CosineDistanceMeasure()));


//    // 计算簇间距离
//    List<Cluster> models = classifier.getModels();
//    //
//    DistanceMeasure measure = new CosineDistanceMeasure();
//    double max = 0;
//    double min = Double.MAX_VALUE;
//    double sum = 0;
//    int count = 0;
//        for (int i = 0; i < models.size(); ++i) {
//        for (int j = 0; j < models.size(); ++j) {
//            double d = measure.distance(models.get(i).getCenter(), models.get(j).getCenter());
//            min = Math.min(d, min);
//            max = Math.max(d, max);
//            sum += d;
//            ++count;
//        }
//    }
//        System.out.printf(
//                "max distance : %f, min distance: % f avg distance: %f \n",
//    max ,
//    min,
//            (sum / count - min / (max - min))
//            );

    /**
     * @param word
     * @param dstDoc
     * @param tokenMap
     * @return
     */
    static double calcTFIDF(String word, Multiset<String> dstDoc, HashMap<String, Multiset<String>> tokenMap) {
        double tf = dstDoc.count(word) * 1.0 / dstDoc.size();
        double idf = 0;
        int containDoc = 0;
        for (Map.Entry<String, Multiset<String>> entry : tokenMap.entrySet()) {
            // 3.1 因为中文分词，分出来的值不是连续的，使用RandomAccessSparseVector最合适
            //     这里设置为1000，只有1000个分词起作用

            Multiset<String> words = entry.getValue();
            if (words.contains(word)) {
                ++containDoc;
            }
        }
        idf = Math.log(tokenMap.size() * 1.0 / (containDoc + 1));
        return tf * idf;
    }
}