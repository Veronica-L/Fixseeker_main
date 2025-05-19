import io.shiftleft.semanticcpg.language._
import upickle.default.*
import os.Path

@main def exec(cpgFile: String, FuncStr: String, outpath: String) = {
    importCode(cpgFile)
    case class MethodInfo(id: Long, code: String, filename: String, fullName: String, isExternal: Boolean, lineNumber: Integer, lineNumberEnd: Integer, name: String, signature: String, caller: List[Long], callee: List[Long]) derives ReadWriter
    case class FileMethods(methodList: List[MethodInfo]) derives ReadWriter
    val prefixes = Array("<operator>", "<includes>")

    var FuncList: Array[String] = Array()
    if (FuncStr.contains(",")){
        FuncList = FuncStr.split(",")
    }
    else{
        FuncList = Array(FuncStr)
    }
    var ml: List[MethodInfo] = List()
    for( m <- cpg.method ){
        val methodname = m.name
        for (fun <- FuncList){
            if (methodname==fun){
                println(methodname)
                val startsWithAnyPrefix = prefixes.exists(prefix => methodname.startsWith(prefix))
                if (!startsWithAnyPrefix) {
                    println(methodname)
                    var caller_list: List[Long] = List()
                    var callee_list: List[Long] = List()
                    var linenumber: Integer = new Integer(0)
                    var linenumberend: Integer = new Integer(0)
                    for (caller <- m.caller){
                        caller_list = caller_list :+ caller.id
                    }

                    for (callee <- m.callee ){
                        callee_list = callee_list :+ callee.id
                    }
                    if (m.lineNumber != None){
                        linenumber = m.lineNumber.head
                    }
                    if (m.lineNumberEnd != None){
                        linenumberend = m.lineNumberEnd.head
                    }
                    val mi = MethodInfo(id=m.id, code=m.code, filename=m.filename, fullName=m.name, isExternal=m.isExternal, lineNumber=linenumber, lineNumberEnd=linenumberend, name=m.name, signature=m.signature, caller=caller_list, callee=callee_list)
                    ml = ml:+ mi
                }
            }
        }
    }
    val fms = FileMethods(methodList=ml)
    val json: String = write(fms)

    val p: Path = os.Path(outpath)
    os.write(p, json)
}