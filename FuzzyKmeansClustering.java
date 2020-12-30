package com.digquant;

import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.canopy.Canopy;
import org.apache.mahout.clustering.canopy.CanopyClusterer;
import org.apache.mahout.clustering.classify.ClusterClassifier;
import org.apache.mahout.clustering.fuzzykmeans.SoftCluster;
import org.apache.mahout.clustering.iterator.FuzzyKMeansClusteringPolicy;
import org.apache.mahout.clustering.iterator.KMeansClusteringPolicy;
import org.apache.mahout.clustering.kmeans.Kluster;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;


import java.util.ArrayList;
import java.util.List;

public class FuzzyKmeansClustering {


    /**
     * 加载数据
     *
     * @return
     */
    static public List<Vector> loadData() {
        List<Vector> sampleData = new ArrayList<>();
        // 使用数据构建
        RandomPointsUtil.generateSamples(sampleData, 20, 1, 1, 3);
//        RandomPointsUtil.generateSamples(sampleData, 300, 1, 0, 0.5);
//        RandomPointsUtil.generateSamples(sampleData, 300, 0, 2, 0.1);

        return sampleData;
    }

    public static void main(String[] args) {
        // 随机生成的点
        List<Vector> sampleData = loadData();

        List<Vector> sampleData2 = new ArrayList<>(sampleData);

        List<Canopy> canopies = CanopyClusterer.createCanopies(sampleData, new EuclideanDistanceMeasure(), 3.0, 1.5);

        List<Cluster> clusterModels = new ArrayList<>();
        int clusterId = 0;
        for (Canopy canopy : canopies) {
            Vector vec = canopy.getCenter();
            clusterModels.add(new SoftCluster(vec, clusterId++, new EuclideanDistanceMeasure()));
        }
        // 3，构建聚类算法
        FuzzyKMeansClusteringPolicy fuzzyKMeansClusteringPolicy = new FuzzyKMeansClusteringPolicy();
        ClusterClassifier classifier = new ClusterClassifier(clusterModels, fuzzyKMeansClusteringPolicy);
        fuzzyKMeansClusteringPolicy.update(classifier);
        // 4， 运行算法，这里默认迭代10次
        for (int i = 0; i < 100; ++i) {
            for (Vector vector : sampleData2) {
                Vector probabilities = classifier.classify(vector);
                Vector weights = fuzzyKMeansClusteringPolicy.select(probabilities);
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
        for (Vector vector : sampleData2) {
            Vector pdfPerCluster = classifier.classify(vector);
            if (pdfPerCluster.maxValue() >= 0) {
                System.out.printf("clusterId = %d, point(%f, %f)\n", pdfPerCluster.maxValueIndex(), vector.get(0),vector.get(1));
            }
        }

    }
}
