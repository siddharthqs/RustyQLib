use csv;
pub fn read_ts(path: &str){
    let mut reader = csv::Reader::from_path(path).unwrap();
    for record in reader.records() {
        let r = record.unwrap();
        println!("{:?}", &r[0]);
        println!("{:?}", &r[1]);

    }

}
