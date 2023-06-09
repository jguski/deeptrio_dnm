//////////////////////////////////////////////////////////////////////
// This file was automatically generated by PyCLIF.
// Version 0.3
//////////////////////////////////////////////////////////////////////
// source: third_party/nucleus/io/python/reference.clif

#include <memory>
#include "absl/types/optional.h"
#include "third_party/nucleus/io/reference.h"
#include "clif/python/postconv.h"

namespace nucleus {
using namespace ::clif;

// CLIF use `::nucleus::GenomeReference` as GenomeReference
bool Clif_PyObjAs(PyObject* input, ::nucleus::GenomeReference** output);
bool Clif_PyObjAs(PyObject* input, std::shared_ptr<::nucleus::GenomeReference>* output);
bool Clif_PyObjAs(PyObject* input, std::unique_ptr<::nucleus::GenomeReference>* output);
PyObject* Clif_PyObjFrom(::nucleus::GenomeReference*, py::PostConv);
PyObject* Clif_PyObjFrom(std::shared_ptr<::nucleus::GenomeReference>, py::PostConv);
PyObject* Clif_PyObjFrom(std::unique_ptr<::nucleus::GenomeReference>, py::PostConv);
template<typename T>
typename std::enable_if<std::is_same<T, ::nucleus::GenomeReference>::value>::type Clif_PyObjFrom(const ::nucleus::GenomeReference*, py::PostConv) = delete;
template<typename T>
typename std::enable_if<std::is_same<T, ::nucleus::GenomeReference>::value>::type Clif_PyObjFrom(const ::nucleus::GenomeReference&, py::PostConv) = delete;
// CLIF use `::nucleus::Iterable<std::pair< ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>, ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>>>` as GenomeReferenceRecordIterable
bool Clif_PyObjAs(PyObject* input, ::nucleus::Iterable<std::pair< ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>, ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>>>** output);
bool Clif_PyObjAs(PyObject* input, std::shared_ptr<::nucleus::Iterable<std::pair< ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>, ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>>>>* output);
bool Clif_PyObjAs(PyObject* input, std::unique_ptr<::nucleus::Iterable<std::pair< ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>, ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>>>>* output);
PyObject* Clif_PyObjFrom(::nucleus::Iterable<std::pair< ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>, ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>>>*, py::PostConv);
PyObject* Clif_PyObjFrom(std::shared_ptr<::nucleus::Iterable<std::pair< ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>, ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>>>>, py::PostConv);
PyObject* Clif_PyObjFrom(std::unique_ptr<::nucleus::Iterable<std::pair< ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>, ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>>>>, py::PostConv);
template<typename T>
typename std::enable_if<std::is_same<T, ::nucleus::Iterable<std::pair< ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>, ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>>>>::value>::type Clif_PyObjFrom(const ::nucleus::Iterable<std::pair< ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>, ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>>>*, py::PostConv) = delete;
template<typename T>
typename std::enable_if<std::is_same<T, ::nucleus::Iterable<std::pair< ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>, ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>>>>::value>::type Clif_PyObjFrom(const ::nucleus::Iterable<std::pair< ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>, ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>>>&, py::PostConv) = delete;
// CLIF use `::nucleus::InMemoryFastaReader` as InMemoryFastaReader
bool Clif_PyObjAs(PyObject* input, ::nucleus::InMemoryFastaReader** output);
bool Clif_PyObjAs(PyObject* input, std::shared_ptr<::nucleus::InMemoryFastaReader>* output);
bool Clif_PyObjAs(PyObject* input, std::unique_ptr<::nucleus::InMemoryFastaReader>* output);
PyObject* Clif_PyObjFrom(::nucleus::InMemoryFastaReader*, py::PostConv);
PyObject* Clif_PyObjFrom(std::shared_ptr<::nucleus::InMemoryFastaReader>, py::PostConv);
PyObject* Clif_PyObjFrom(std::unique_ptr<::nucleus::InMemoryFastaReader>, py::PostConv);
template<typename T>
typename std::enable_if<std::is_same<T, ::nucleus::InMemoryFastaReader>::value>::type Clif_PyObjFrom(const ::nucleus::InMemoryFastaReader*, py::PostConv) = delete;
template<typename T>
typename std::enable_if<std::is_same<T, ::nucleus::InMemoryFastaReader>::value>::type Clif_PyObjFrom(const ::nucleus::InMemoryFastaReader&, py::PostConv) = delete;
// CLIF use `::nucleus::IndexedFastaReader` as IndexedFastaReader
bool Clif_PyObjAs(PyObject* input, ::nucleus::IndexedFastaReader** output);
bool Clif_PyObjAs(PyObject* input, std::shared_ptr<::nucleus::IndexedFastaReader>* output);
bool Clif_PyObjAs(PyObject* input, std::unique_ptr<::nucleus::IndexedFastaReader>* output);
PyObject* Clif_PyObjFrom(::nucleus::IndexedFastaReader*, py::PostConv);
PyObject* Clif_PyObjFrom(std::shared_ptr<::nucleus::IndexedFastaReader>, py::PostConv);
PyObject* Clif_PyObjFrom(std::unique_ptr<::nucleus::IndexedFastaReader>, py::PostConv);
template<typename T>
typename std::enable_if<std::is_same<T, ::nucleus::IndexedFastaReader>::value>::type Clif_PyObjFrom(const ::nucleus::IndexedFastaReader*, py::PostConv) = delete;
template<typename T>
typename std::enable_if<std::is_same<T, ::nucleus::IndexedFastaReader>::value>::type Clif_PyObjFrom(const ::nucleus::IndexedFastaReader&, py::PostConv) = delete;
// CLIF use `::nucleus::UnindexedFastaReader` as UnindexedFastaReader
bool Clif_PyObjAs(PyObject* input, ::nucleus::UnindexedFastaReader** output);
bool Clif_PyObjAs(PyObject* input, std::shared_ptr<::nucleus::UnindexedFastaReader>* output);
bool Clif_PyObjAs(PyObject* input, std::unique_ptr<::nucleus::UnindexedFastaReader>* output);
PyObject* Clif_PyObjFrom(::nucleus::UnindexedFastaReader*, py::PostConv);
PyObject* Clif_PyObjFrom(std::shared_ptr<::nucleus::UnindexedFastaReader>, py::PostConv);
PyObject* Clif_PyObjFrom(std::unique_ptr<::nucleus::UnindexedFastaReader>, py::PostConv);
template<typename T>
typename std::enable_if<std::is_same<T, ::nucleus::UnindexedFastaReader>::value>::type Clif_PyObjFrom(const ::nucleus::UnindexedFastaReader*, py::PostConv) = delete;
template<typename T>
typename std::enable_if<std::is_same<T, ::nucleus::UnindexedFastaReader>::value>::type Clif_PyObjFrom(const ::nucleus::UnindexedFastaReader&, py::PostConv) = delete;

}  // namespace nucleus

// CLIF init_module if (PyObject* m = PyImport_ImportModule("third_party.nucleus.io.python.reference")) Py_DECREF(m);
// CLIF init_module else goto err;
