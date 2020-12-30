package com.digquant;

import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.ClusterClassifier;
import org.apache.mahout.clustering.iterator.KMeansClusteringPolicy;
import org.apache.mahout.clustering.kmeans.Kluster;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.List;

public class ClusterApple {

    /**
     * 加载数据
     *
     * @return
     */
    static public List<NamedVector> loadData() {
        List<NamedVector> apples = new ArrayList<>();
        NamedVector apple1 = new NamedVector(new DenseVector(new double[]{0.11, 510, 1}), "Small round green apple");
        apples.add(apple1);

        NamedVector apple2 = new NamedVector(new DenseVector(new double[] {0.23, 650, 3}), "Large oval red apple");
        apples.add(apple2);

        NamedVector apple3 = new NamedVector(new DenseVector(new double[] {0.09, 630, 1}), "Small elongated red apple");
        apples.add(apple3);

        NamedVector apple4 = new NamedVector(new DenseVector(new double[] {0.25, 590, 3}), "Large round yellow apple");
        apples.add(apple4);

        NamedVector apple5 = new NamedVector(
                new DenseVector(new double[] {0.18, 520, 2}),
                "Medium oval green apple");
        apples.add(apple5);
        return apples;
    }

    public static void main(String[] args) {
        // 1, 加载数据
        List<NamedVector> apples = loadData();
        // 2，设置初始的簇中心，并且设置计算点到簇中心距离的算法
        List<Cluster> clusterModels = new ArrayList<>();
        for (int i = 0; i < 4; ++i) {
            Vector vec = apples.get(i);
            clusterModels.add(new Kluster(vec, i, new EuclideanDistanceMeasure()));
        }
        // 3，构建聚类算法
        KMeansClusteringPolicy kMeansClusteringPolicy = new KMeansClusteringPolicy();
        ClusterClassifier classifier = new ClusterClassifier(clusterModels, kMeansClusteringPolicy);
        kMeansClusteringPolicy.update(classifier);
        // 4， 运行算法，这里默认迭代10次
        for (int i = 0; i < 10; ++i) {
            for (NamedVector vector : apples) {
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
            for (Cluster cluster:  models) {
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
        for (NamedVector vector : apples) {
            Vector pdfPerCluster = classifier.classify(vector);
            if (pdfPerCluster.maxValue() >= 0) {
                System.out.printf("name = %s clusterId = %d, weight=%f, color = %f, size = %s\n", vector.getName() ,pdfPerCluster.maxValueIndex(), vector.get(0), vector.get(1), vector.get(2));
            }
        }



    }

}
