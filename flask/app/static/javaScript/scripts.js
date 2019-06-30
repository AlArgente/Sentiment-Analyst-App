function myFunction() {
  // var x = document.getElementById("ohsnap");
  alert("I've been clicked!!");
  console.log('Va');
}


function trainAlgValidateForm() {
  var x = document.forms["form-prepro"]["algoritmo"].value;
  if (x == "") {
    alert("Tiene que seleccionar un algoritmo");
  }
}

function trainPreproValidateForm() {
  var e = document.getElementById["preprolist"];
}

function greeting(alg) {
  alert('Submit button clicked!!');
  return true;
}


function preproTrainValidateForm() {
  var x = document.querySelector('.loader');
  x.setAttribute('style', 'display: flex');
  console.log('Llego aquí');
  document.getElementById("form-prepo").submit();
}


function nuevaFunction() {
  var re = document.getElementById('results_test');
  // re.style.display = "none";
  if (re.style.display == "none") {
    re.style.display = "block";
  } else {
    re.style.display = "none";
  }
}

function loading() {
  var x = document.querySelector('.loader');
  x.setAttribute('style', 'display: flex');
  console.log('Llego aquí');
}
