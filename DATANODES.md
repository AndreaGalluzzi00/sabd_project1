# Cluster HDFS: 1 datanode (timing) vs 3 datanode (demo replica)

Il progetto gira **solo in locale** (Mac M1, Colima/QEMU). Due modalità.

## Modalità 1 — Benchmark / timing (default)

1 datanode, `dfs.replication=1`. È la configurazione da usare per **tutte le
misure di tempo** (Q1 baseline ~152s): nessun overhead di replica, risultati
puliti e riproducibili.

```bash
docker compose up -d
```

## Modalità 2 — Demo replica automatica (3 datanode)

Aggiunge `datanode2` e `datanode3` tramite l'override. Serve solo per
**mostrare/documentare** la replica e la fault tolerance, NON per i timing.

```bash
docker compose -f docker-compose.yml -f docker-compose.datanodes.yml up -d
```

Verifica che i 3 datanode siano registrati (Live nodes = 3):

```bash
docker exec namenode hdfs dfsadmin -report | grep -E 'Live datanodes|Name:'
# oppure dalla Web UI: http://localhost:9870  -> Datanodes
```

### Impostare la replica a 3

La replica di un file la decide il **client** al momento della scrittura
(`dfs.replication` in `nifi/config/hdfs-site.xml`, attualmente 1). Per la demo
hai due strade:

- **Rapida (su file già esistenti):**
  ```bash
  docker exec namenode hdfs dfs -setrep -R -w 3 /
  ```
- **Permanente (file scritti dopo):** porta a `3` il valore di
  `dfs.replication` in `nifi/config/hdfs-site.xml` (client Spark/NiFi) e in
  `hadoop.env` (`HDFS_CONF_dfs_replication=3`, default del namenode), poi
  riscrivi i dati. ⚠️ Ricordati di riportarlo a `1` prima dei benchmark.

### Mostrare la replica/fault tolerance

```bash
# dove stanno i blocchi di un file
docker exec namenode hdfs fsck /percorso/del/file -files -blocks -locations

# spegni un datanode e osserva la ri-replica automatica
docker stop datanode3
docker exec namenode hdfs dfsadmin -report   # Live datanodes scende a 2
# il namenode ri-replica i blocchi sotto-replicati sui nodi rimasti
```

> Regola: il fattore di replica non può superare il numero di datanode vivi.
> Con 3 datanode puoi usare replica 1, 2 o 3.

## Pulizia

```bash
# ferma tutto (i volumi datanode2/3 restano)
docker compose -f docker-compose.yml -f docker-compose.datanodes.yml down

# rimuovi anche i volumi extra dei datanode di demo
docker volume rm sabd_project1_hdfs_datanode2 sabd_project1_hdfs_datanode3
```
