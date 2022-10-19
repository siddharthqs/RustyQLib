//Algorithm for computing natural cubic splines
// https://en.wikipedia.org/w/index.php?title=Spline_%28mathematics%29&oldid=288288033#Algorithm_for_computing_natural_cubic_splines
//https://stackoverflow.com/questions/1204553/are-there-any-good-libraries-for-solving-cubic-splines-in-c

pub struct CubicSpline<'a>{
    pub x_vec : &'a Vec<f64>,
    pub y_vec : &'a Vec<f64>,
    pub spline_set : Vec<SplineSet>
}

pub struct SplineSet{
    a:f64,
    b:f64,
    c:f64,
    d:f64,
    x:f64,
}
impl CubicSpline<'_> {
    pub fn new<'a>(x: &'a Vec<f64>, y: &'a Vec<f64>) -> CubicSpline<'a> {
        assert!(x.len() == y.len() && x.len() >= 2 && y.len() >= 2, "Must have at least 2 control points.");
        let n = x.len()-1;
        let mut a = vec![x[0]];
        let mut aa = y.clone();
        a.append(&mut aa);
        let mut c = vec![0.0; n+1];
        let mut b = vec![0.0; n];
        let mut d = vec![0.0; n];
        let mut h = Vec::new();
        for i in 0..n {
            h.push(x[i+1]-x[i]);
        }
        println!("h: {:?}", h);
        let mut nu = Vec::new();
        for i in 0..n {
            nu.push(y[i+1]-y[i]);
        }

        println!("nu: {:?}", nu);
        let mut alpha = vec![0.0];
        for i in 1..n {
            alpha.push(3.0*(nu[i]/h[i] - nu[i-1]/h[i-1]));
        }
        println!("alpha: {:?}", alpha);
        let mut l = vec![0.0; n+1];
        let mut mu = vec![0.0; n+1];
        let mut z = vec![0.0; n+1];
        l[0] = 1.0;
        for i in 1..n{
            l[i] = 2.0*(x[i+1]-x[i-1]) - h[i-1]*mu[i-1];
            mu[i] = h[i]/l[i];
            z[i] = (alpha[i]-h[i-1]*z[i-1])/l[i];
        }
        l[n] = 1.0;
        z[n] = 0.0;
        //c[n] = 0;
        for j in (0..n).rev(){
            c[j] = z[j] - mu[j] * c[j+1];
            b[j] = (a[j+1]-a[j])/h[j]-h[j]*(c[j+1]+2.0*c[j])/3.0;
            d[j] = (c[j+1]-c[j])/(3.0*h[j]);

        }
        let mut output = Vec::new();
        for i in 0..n {
            let spline_set = SplineSet{
                a:a[i],
                b:b[i],
                c:c[i],
                d:d[i],
                x:x[i]
            };
            output.push(spline_set);
        }
        let spline = CubicSpline{
            x_vec:x,
            y_vec:y,
            spline_set:output
        };
        spline
    }
    pub fn interpolation(&self,x:f64) -> f64 {
        let n = self.x_vec.len();
        for i in 0..n {
            if x<=self.x_vec[i] {
                println!("for");
                let diff = self.x_vec[i] - x;
                println!("{}",diff);
                println!("{}",self.spline_set[i].a);
                println!("{}",self.spline_set[i].b);
                println!("{}",self.spline_set[i].c);
                println!("{}",self.spline_set[i].d);
                return self.spline_set[i].a +
                self.spline_set[i].b * diff +
                self.spline_set[i].c * diff * diff +
                self.spline_set[i].d * diff * diff * diff;
            }
        }
    0.0
    }
}